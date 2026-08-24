"""Solver-independent D-attachment existence scans for Rao B-D-E topology.

This module deliberately stays out of the JAX/Optimistix/least-squares stack.
It samples candidate ``(theta_B, kernel_d_fraction)`` pairs, builds the
NASA-style kernel with the existing pure-Python/NumPy MOC port, starts the
downstream DE characteristic from point D, and records exit residual fields.

The three supported D-attachment closures are:

``smooth``
    Position, flow angle, and Mach number are inherited continuously from the
    kernel point D.
``position``
    Position is inherited from D, while the post-D flow angle and Mach number
    are selected by an explicit local grid scan.  The stored residuals are the
    minimum-norm result over that grid.
``fan``
    Position is inherited from D, while the post-D state is connected to the
    pre-D state by a centered Prandtl-Meyer fan,
    ``nu_post - nu_pre = theta_post - theta_pre``.  The stored residuals are
    the minimum-norm result over the requested fan-turn grid.

The B5 public artifact is not one of those internal ``theta_B x kdf`` slices.
``scan_convergence_domain`` sweeps the external ``epsilon x length_pct`` design
box, uses each internal scan only as a root seed, then records the refined
``theta_B`` and ``kdf`` as evidence outputs.  Its typed, hash-pinned domain
classifies residual convergence, kernel refinement, and Rao Eq. (5) validity
separately.  Numerical failure remains unknown rather than being labelled
physical non-existence, and lookups never extrapolate beyond the declared box.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from bisect import bisect_left
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from raosim.gas_dynamics import (
    isentropic_pressure_ratio,
    mach_from_area_ratio,
    mach_from_prandtl_meyer,
    prandtl_meyer,
    thrust_coefficient,
)
from raosim.nasa_moc import (
    MOCNode,
    build_kernel,
    calc_mdot_bd,
    find_point_e,
    nasa_runge_kutta,
    surface_thrust_coefficient,
)


MODEL_SMOOTH = "smooth"
MODEL_POSITION = "position"
MODEL_FAN = "fan"
MODELS = (MODEL_SMOOTH, MODEL_POSITION, MODEL_FAN)

STOP_MASS = "mass"
STOP_RADIUS = "radius"
STOP_LENGTH = "length"
STOP_MODES = (STOP_MASS, STOP_RADIUS, STOP_LENGTH)

THIRD_AUTO = "auto"
THIRD_MASS = "mass"
THIRD_PERFORMANCE = "performance"
THIRD_RAO_EXIT = "rao_exit"
THIRD_COMPONENTS = (THIRD_AUTO, THIRD_MASS, THIRD_PERFORMANCE, THIRD_RAO_EXIT)

SCAN_GEOMETRY = "geometry"
SCAN_STATIONARITY = "stationarity"
SCAN_DIAGNOSTIC = "diagnostic"
SCAN_MODES = (SCAN_GEOMETRY, SCAN_STATIONARITY, SCAN_DIAGNOSTIC)

FAILURE_CODES: Mapping[int, str] = {
    0: "ok",
    1: "kernel_build_failed",
    2: "invalid_d_point",
    3: "invalid_post_state",
    4: "integration_failed",
    5: "nonfinite_residual",
    6: "no_candidate",
}

CONVERGENCE_DOMAIN_SCHEMA_VERSION = 1
CONVERGENCE_DOMAIN_SOURCE_MODEL_ID = (
    "raosim.rao_existence_scan:epsilon-length-fixed-context-d-attachment-domain:v1"
)

SAMPLE_CONVERGED_VALID = "converged_valid"
SAMPLE_CONVERGED_INVALID = "converged_invalid"
SAMPLE_NOT_CONVERGED = "not_converged"
SAMPLE_NUMERICAL_FAILURE = "numerical_failure"
SAMPLE_UNEVALUABLE = "unevaluable"
SAMPLE_UNDER_RESOLVED = "under_resolved"
SAMPLE_CLASSIFICATIONS = (
    SAMPLE_CONVERGED_VALID,
    SAMPLE_CONVERGED_INVALID,
    SAMPLE_NOT_CONVERGED,
    SAMPLE_NUMERICAL_FAILURE,
    SAMPLE_UNEVALUABLE,
    SAMPLE_UNDER_RESOLVED,
)

DOMAIN_SUPPORTED = "supported"
DOMAIN_REFUSED_INVALID_RAO_REGION = "refused_invalid_rao_region"
DOMAIN_REFUSED_NOT_CONVERGED = "refused_not_converged"
DOMAIN_UNKNOWN_OUTSIDE = "unknown_outside_domain"
DOMAIN_UNKNOWN_UNDER_RESOLVED = "unknown_under_resolved"
DOMAIN_UNKNOWN_SCAN_FAILURE = "unknown_scan_failure"
DOMAIN_CONTEXT_MISMATCH = "context_mismatch"
DOMAIN_MODEL_MISMATCH = "model_mismatch"
DOMAIN_LOOKUP_STATUSES = (
    DOMAIN_SUPPORTED,
    DOMAIN_REFUSED_INVALID_RAO_REGION,
    DOMAIN_REFUSED_NOT_CONVERGED,
    DOMAIN_UNKNOWN_OUTSIDE,
    DOMAIN_UNKNOWN_UNDER_RESOLVED,
    DOMAIN_UNKNOWN_SCAN_FAILURE,
    DOMAIN_CONTEXT_MISMATCH,
    DOMAIN_MODEL_MISMATCH,
)

RESOLUTION_PASSED = "passed"
RESOLUTION_FAILED = "failed"
RESOLUTION_NOT_EVALUATED = "not_evaluated"
RESOLUTION_STATUSES = (
    RESOLUTION_PASSED,
    RESOLUTION_FAILED,
    RESOLUTION_NOT_EVALUATED,
)

THETA_SEED_FIXED = "fixed_axis"
THETA_SEED_RAO_CHART = "rao_chart_centered"
THETA_SEED_POLICIES = (THETA_SEED_FIXED, THETA_SEED_RAO_CHART)
RAO_CHART_THETA_SEED_SOURCE_ID = (
    "raosim.nozzle_geometry.lookup_angles:rao-top-chart:v1"
)


@dataclass(frozen=True)
class ExistenceScanConfig:
    """Configuration for a NumPy-only D-attachment existence scan."""

    Rt: float
    epsilon: float
    gamma: float = 1.4
    pa_over_p0: float = 0.0
    length_pct: float = 80.0
    throat_downstream_radius_factor: float = 0.382
    throat_upstream_radius_factor: float = 1.5
    starting_line_method: str = "kliegel_levine"
    n_kernel: int = 16
    n_de_points: int = 24
    theta_b_values_deg: np.ndarray | None = None
    kdf_values: np.ndarray | None = None
    theta_b_center_deg: float = 24.0
    theta_b_span_deg: float = 10.0
    theta_b_count: int = 21
    kdf_min: float = 0.05
    kdf_max: float = 0.85
    kdf_count: int = 21
    models: tuple[str, ...] = MODELS
    stop_at: str = STOP_MASS
    scan_mode: str = SCAN_GEOMETRY
    third_residual: str = THIRD_AUTO
    residual_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)
    position_theta_span_deg: float = 10.0
    position_theta_count: int = 9
    position_mach_down: float = 0.5
    position_mach_up: float = 1.5
    position_mach_count: int = 9
    fan_turn_min_deg: float = 0.0
    fan_turn_max_deg: float = 12.0
    fan_turn_count: int = 13
    max_mach: float = 12.0
    max_de_steps: int = 240

    @property
    def target_radius(self) -> float:
        return math.sqrt(self.epsilon) * self.Rt

    @property
    def target_length(self) -> float:
        return target_length(self.Rt, self.epsilon, self.length_pct)

    @property
    def theta_axis(self) -> np.ndarray:
        if self.theta_b_values_deg is not None:
            return np.asarray(self.theta_b_values_deg, dtype=float)
        lo = self.theta_b_center_deg - 0.5 * self.theta_b_span_deg
        hi = self.theta_b_center_deg + 0.5 * self.theta_b_span_deg
        return np.linspace(lo, hi, max(int(self.theta_b_count), 1))

    @property
    def kdf_axis(self) -> np.ndarray:
        if self.kdf_values is not None:
            return np.asarray(self.kdf_values, dtype=float)
        return np.linspace(
            float(self.kdf_min),
            float(self.kdf_max),
            max(int(self.kdf_count), 1),
        )


@dataclass
class DEIntegrationResult:
    """Result of one deterministic DE initial-value integration."""

    nodes: tuple[MOCNode, ...]
    mass: float
    success: bool
    failure_code: int = 0

    @property
    def E(self) -> MOCNode | None:
        return self.nodes[-1] if self.nodes else None


@dataclass
class ClosureScanResult:
    """Heatmap arrays for one D-attachment closure."""

    model: str
    theta_b_deg: np.ndarray
    kdf: np.ndarray
    residual_norm: np.ndarray
    radius_residual: np.ndarray
    length_residual: np.ndarray
    sigma_E_rad: np.ndarray
    theta_rao_E_deg: np.ndarray
    mass_residual: np.ndarray
    performance_residual: np.ndarray
    rao_exit_residual: np.ndarray
    thrust_coefficient: np.ndarray
    thrust_coefficient_target: np.ndarray
    d_mach_pre: np.ndarray
    d_mach_post: np.ndarray
    d_mach_jump: np.ndarray
    d_theta_pre_deg: np.ndarray
    d_theta_post_deg: np.ndarray
    d_theta_jump_deg: np.ndarray
    fan_turn_deg: np.ndarray
    exit_angle_deg: np.ndarray
    exit_mach: np.ndarray
    rao_eq5_evaluable: np.ndarray
    rao_eq5_valid: np.ndarray
    rao_eq5_min_dr_over_r: np.ndarray
    success: np.ndarray
    failure_code: np.ndarray

    def best_index(self) -> tuple[int, int]:
        """Return the ``(theta_index, kdf_index)`` of the lowest norm."""
        finite = np.where(np.isfinite(self.residual_norm), self.residual_norm, np.inf)
        flat = int(np.argmin(finite))
        return tuple(int(v) for v in np.unravel_index(flat, self.residual_norm.shape))

    def best_summary(self) -> dict:
        """Compact scalar summary of the best sampled cell."""
        i, j = self.best_index()
        code = int(self.failure_code[i, j])
        return {
            "model": self.model,
            "theta_B_deg": float(self.theta_b_deg[i]),
            "kdf": float(self.kdf[j]),
            "residual_norm": float(self.residual_norm[i, j]),
            "radius_residual": float(self.radius_residual[i, j]),
            "length_residual": float(self.length_residual[i, j]),
            "sigma_E_rad": float(self.sigma_E_rad[i, j]),
            "theta_Rao_E_deg": float(self.theta_rao_E_deg[i, j]),
            "mass_residual": float(self.mass_residual[i, j]),
            "performance_residual": float(self.performance_residual[i, j]),
            "rao_exit_residual": float(self.rao_exit_residual[i, j]),
            "thrust_coefficient": float(self.thrust_coefficient[i, j]),
            "thrust_coefficient_target": float(self.thrust_coefficient_target[i, j]),
            "D_M_pre": float(self.d_mach_pre[i, j]),
            "D_M_post": float(self.d_mach_post[i, j]),
            "D_theta_pre_deg": float(self.d_theta_pre_deg[i, j]),
            "D_theta_post_deg": float(self.d_theta_post_deg[i, j]),
            "fan_turn_deg": float(self.fan_turn_deg[i, j]),
            "exit_angle_deg": float(self.exit_angle_deg[i, j]),
            "exit_mach": float(self.exit_mach[i, j]),
            "rao_eq5_evaluable": bool(self.rao_eq5_evaluable[i, j]),
            "rao_eq5_valid": bool(self.rao_eq5_valid[i, j]),
            "rao_eq5_min_dr_over_r": float(
                self.rao_eq5_min_dr_over_r[i, j]
            ),
            "success": bool(self.success[i, j]),
            "failure_code": code,
            "failure_reason": FAILURE_CODES.get(code, "unknown"),
        }


@dataclass
class ExistenceScanResult:
    """Full multi-closure scan result."""

    config: ExistenceScanConfig
    closures: dict[str, ClosureScanResult] = field(default_factory=dict)

    def summary(self) -> dict:
        """JSON-friendly summary of all closure minima."""
        return {
            "config": {
                "Rt": self.config.Rt,
                "epsilon": self.config.epsilon,
                "gamma": self.config.gamma,
                "pa_over_p0": self.config.pa_over_p0,
                "length_pct": self.config.length_pct,
                "target_radius": self.config.target_radius,
                "target_length": self.config.target_length,
                "starting_line_method": self.config.starting_line_method,
                "n_kernel": self.config.n_kernel,
                "n_de_points": self.config.n_de_points,
                "max_de_steps": self.config.max_de_steps,
                "stop_at": self.config.stop_at,
                "scan_mode": self.config.scan_mode,
                "third_residual": self.config.third_residual,
                "theta_B_count": int(len(self.config.theta_axis)),
                "kdf_count": int(len(self.config.kdf_axis)),
                "models": list(self.config.models),
            },
            "best": {
                name: closure.best_summary()
                for name, closure in self.closures.items()
            },
        }


@dataclass
class RootRefineResult:
    """Geometry root refinement result for one closure."""

    model: str
    success: bool
    message: str
    n_kernel: int
    theta_B_deg: float
    kdf: float
    fan_turn_deg: float
    radius_residual: float
    length_residual: float
    sigma_E_rad: float
    mass_residual: float
    performance_residual: float
    rao_exit_residual: float
    residual_norm_geometry: float
    residual_norm_stationarity: float
    D_M_pre: float
    D_M_post: float
    D_theta_pre_deg: float
    D_theta_post_deg: float
    exit_angle_deg: float
    exit_mach: float

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "success": bool(self.success),
            "message": self.message,
            "n_kernel": int(self.n_kernel),
            "theta_B_deg": float(self.theta_B_deg),
            "kdf": float(self.kdf),
            "fan_turn_deg": float(self.fan_turn_deg),
            "radius_residual": float(self.radius_residual),
            "length_residual": float(self.length_residual),
            "sigma_E_rad": float(self.sigma_E_rad),
            "mass_residual": float(self.mass_residual),
            "performance_residual": float(self.performance_residual),
            "rao_exit_residual": float(self.rao_exit_residual),
            "residual_norm_geometry": float(self.residual_norm_geometry),
            "residual_norm_stationarity": float(self.residual_norm_stationarity),
            "D_M_pre": float(self.D_M_pre),
            "D_M_post": float(self.D_M_post),
            "D_theta_pre_deg": float(self.D_theta_pre_deg),
            "D_theta_post_deg": float(self.D_theta_post_deg),
            "exit_angle_deg": float(self.exit_angle_deg),
            "exit_mach": float(self.exit_mach),
        }


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return strict, deterministic JSON bytes for an artifact payload."""
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _json_safe(value: Any) -> Any:
    """Normalize NumPy/scalar scan data without non-standard JSON numbers."""
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if value is None or isinstance(value, str):
        return value
    raise TypeError(f"cannot serialize scan value of type {type(value).__name__}")


def _optional_finite_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _require_finite(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _rao_eq5_source_id() -> str:
    """Return B4's canonical Rao Eq. (5) authority without duplicating it."""
    from raosim.rao_variational import RAO_RADIAL_INCREMENT_SOURCE_ID

    return RAO_RADIAL_INCREMENT_SOURCE_ID


@dataclass(frozen=True)
class RaoConvergenceContext:
    """Fixed design and numerical identity for an ``epsilon x L%`` map.

    ``epsilon`` and ``length_pct`` are deliberately absent because they are
    the two public lookup axes.  Every other quantity that changes the scan is
    pinned here.  A lookup requires an expected context and fails closed on an
    exact mismatch; support from one design/mesh context cannot leak to
    another one.
    """

    Rt_reference: float
    gamma: float
    pa_over_p0: float
    throat_downstream_radius_factor: float
    throat_upstream_radius_factor: float
    starting_line_method: str
    n_kernel_finest: int
    n_kernel_ladder: tuple[int, ...]
    n_de_points: int
    max_de_steps: int
    root_maxfev: int
    theta_b_seed_policy: str
    theta_b_seed_axis_deg: tuple[float, ...]
    theta_b_chart_span_deg: float
    theta_b_chart_count: int
    theta_b_chart_source_id: str
    kdf_seed_axis: tuple[float, ...]
    stop_at: str
    scan_mode: str
    third_residual: str
    residual_weights: tuple[float, float, float]
    max_mach: float
    closure_model: str
    backend: str = "numpy_host"
    rt_scale_invariant: bool = True

    def __post_init__(self) -> None:
        for name in (
            "Rt_reference",
            "gamma",
            "pa_over_p0",
            "throat_downstream_radius_factor",
            "throat_upstream_radius_factor",
            "max_mach",
        ):
            _require_finite(name, getattr(self, name))
        if self.Rt_reference <= 0.0:
            raise ValueError("Rt_reference must be positive")
        if self.gamma <= 1.0:
            raise ValueError("gamma must be > 1")
        if self.throat_downstream_radius_factor <= 0.0:
            raise ValueError("throat_downstream_radius_factor must be positive")
        if self.throat_upstream_radius_factor <= 0.0:
            raise ValueError("throat_upstream_radius_factor must be positive")
        if self.n_kernel_finest < 5:
            raise ValueError("n_kernel_finest must be >= 5")
        if not self.n_kernel_ladder:
            raise ValueError("n_kernel_ladder must be non-empty")
        if tuple(sorted(set(self.n_kernel_ladder))) != self.n_kernel_ladder:
            raise ValueError("n_kernel_ladder must be strictly increasing")
        if any(value < 5 for value in self.n_kernel_ladder):
            raise ValueError("all n_kernel_ladder values must be >= 5")
        if self.n_kernel_ladder[-1] != self.n_kernel_finest:
            raise ValueError("n_kernel_ladder must end at n_kernel_finest")
        if self.n_de_points < 2:
            raise ValueError("n_de_points must be >= 2")
        if self.max_de_steps < 1:
            raise ValueError("max_de_steps must be positive")
        if self.root_maxfev < 1:
            raise ValueError("root_maxfev must be positive")
        _require_finite(
            "theta_b_chart_span_deg", self.theta_b_chart_span_deg
        )
        if self.theta_b_seed_policy not in THETA_SEED_POLICIES:
            raise ValueError(
                f"theta_b_seed_policy must be one of {THETA_SEED_POLICIES}"
            )
        if self.theta_b_seed_policy == THETA_SEED_FIXED:
            if not self.theta_b_seed_axis_deg:
                raise ValueError(
                    "fixed_axis theta seed policy requires an explicit axis"
                )
            if self.theta_b_chart_count != 0 or self.theta_b_chart_span_deg != 0.0:
                raise ValueError(
                    "fixed_axis theta seed policy cannot carry chart controls"
                )
            if self.theta_b_chart_source_id:
                raise ValueError(
                    "fixed_axis theta seed policy cannot carry a chart source"
                )
        else:
            if self.theta_b_seed_axis_deg:
                raise ValueError(
                    "chart-centered theta seed policy cannot carry one fixed axis"
                )
            if self.theta_b_chart_count < 2:
                raise ValueError("theta_b_chart_count must be >= 2")
            if self.theta_b_chart_span_deg <= 0.0:
                raise ValueError("theta_b_chart_span_deg must be positive")
            if (
                self.theta_b_chart_source_id
                != RAO_CHART_THETA_SEED_SOURCE_ID
            ):
                raise ValueError("unknown Rao chart theta seed source")
        for name, axis in (
            ("theta_b_seed_axis_deg", self.theta_b_seed_axis_deg),
            ("kdf_seed_axis", self.kdf_seed_axis),
        ):
            if name == "theta_b_seed_axis_deg" and not axis:
                continue
            if not axis:
                raise ValueError(f"{name} must be non-empty")
            for index, value in enumerate(axis):
                _require_finite(f"{name}[{index}]", value)
            if any(right <= left for left, right in zip(axis[:-1], axis[1:])):
                raise ValueError(
                    f"{name} must be strictly increasing with no duplicates"
                )
        if self.stop_at not in STOP_MODES:
            raise ValueError(f"stop_at must be one of {STOP_MODES}")
        if self.scan_mode not in SCAN_MODES:
            raise ValueError(f"scan_mode must be one of {SCAN_MODES}")
        if self.third_residual not in THIRD_COMPONENTS:
            raise ValueError(
                f"third_residual must be one of {THIRD_COMPONENTS}"
            )
        if self.closure_model not in MODELS:
            raise ValueError(f"closure_model must be one of {MODELS}")
        if len(self.residual_weights) != 3:
            raise ValueError("residual_weights must contain three values")
        for index, value in enumerate(self.residual_weights):
            _require_finite(f"residual_weights[{index}]", value)
        if not self.starting_line_method:
            raise ValueError("starting_line_method must be non-empty")
        if self.backend != "numpy_host":
            raise ValueError("the declared B5 domain backend must be numpy_host")
        if not self.rt_scale_invariant:
            raise ValueError(
                "schema v1 is defined only for the nondimensional, Rt-scale-invariant scan"
            )

    @classmethod
    def from_scan_config(
        cls,
        config: ExistenceScanConfig,
        *,
        closure_model: str,
        n_kernel_finest: int | None = None,
        n_kernel_ladder: Sequence[int] | None = None,
        root_maxfev: int = 80,
        theta_b_seed_policy: str = THETA_SEED_FIXED,
        theta_b_chart_span_deg: float = 0.0,
        theta_b_chart_count: int = 0,
    ) -> "RaoConvergenceContext":
        return cls(
            Rt_reference=float(config.Rt),
            gamma=float(config.gamma),
            pa_over_p0=float(config.pa_over_p0),
            throat_downstream_radius_factor=float(
                config.throat_downstream_radius_factor
            ),
            throat_upstream_radius_factor=float(
                config.throat_upstream_radius_factor
            ),
            starting_line_method=str(config.starting_line_method),
            n_kernel_finest=int(
                config.n_kernel
                if n_kernel_finest is None
                else n_kernel_finest
            ),
            n_kernel_ladder=tuple(
                int(value)
                for value in (
                    (config.n_kernel,)
                    if n_kernel_ladder is None
                    else n_kernel_ladder
                )
            ),
            n_de_points=int(config.n_de_points),
            max_de_steps=int(config.max_de_steps),
            root_maxfev=int(root_maxfev),
            theta_b_seed_policy=str(theta_b_seed_policy),
            theta_b_seed_axis_deg=(
                tuple(float(value) for value in config.theta_axis)
                if theta_b_seed_policy == THETA_SEED_FIXED
                else ()
            ),
            theta_b_chart_span_deg=float(theta_b_chart_span_deg),
            theta_b_chart_count=int(theta_b_chart_count),
            theta_b_chart_source_id=(
                RAO_CHART_THETA_SEED_SOURCE_ID
                if theta_b_seed_policy == THETA_SEED_RAO_CHART
                else ""
            ),
            kdf_seed_axis=tuple(float(value) for value in config.kdf_axis),
            stop_at=str(config.stop_at),
            scan_mode=str(config.scan_mode),
            third_residual=str(config.third_residual),
            residual_weights=tuple(
                float(value) for value in config.residual_weights
            ),
            max_mach=float(config.max_mach),
            closure_model=str(closure_model),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "Rt_reference": float(self.Rt_reference),
            "backend": self.backend,
            "closure_model": self.closure_model,
            "gamma": float(self.gamma),
            "max_de_steps": int(self.max_de_steps),
            "max_mach": float(self.max_mach),
            "n_de_points": int(self.n_de_points),
            "n_kernel_finest": int(self.n_kernel_finest),
            "n_kernel_ladder": [int(value) for value in self.n_kernel_ladder],
            "pa_over_p0": float(self.pa_over_p0),
            "residual_weights": [float(v) for v in self.residual_weights],
            "rt_scale_invariant": bool(self.rt_scale_invariant),
            "root_maxfev": int(self.root_maxfev),
            "scan_mode": self.scan_mode,
            "starting_line_method": self.starting_line_method,
            "stop_at": self.stop_at,
            "third_residual": self.third_residual,
            "theta_b_chart_count": int(self.theta_b_chart_count),
            "theta_b_chart_source_id": self.theta_b_chart_source_id,
            "theta_b_chart_span_deg": float(self.theta_b_chart_span_deg),
            "theta_b_seed_axis_deg": list(self.theta_b_seed_axis_deg),
            "theta_b_seed_policy": self.theta_b_seed_policy,
            "throat_downstream_radius_factor": float(
                self.throat_downstream_radius_factor
            ),
            "throat_upstream_radius_factor": float(
                self.throat_upstream_radius_factor
            ),
            "kdf_seed_axis": list(self.kdf_seed_axis),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RaoConvergenceContext":
        return cls(
            Rt_reference=float(payload["Rt_reference"]),
            gamma=float(payload["gamma"]),
            pa_over_p0=float(payload["pa_over_p0"]),
            throat_downstream_radius_factor=float(
                payload["throat_downstream_radius_factor"]
            ),
            throat_upstream_radius_factor=float(
                payload["throat_upstream_radius_factor"]
            ),
            starting_line_method=str(payload["starting_line_method"]),
            n_kernel_finest=int(payload["n_kernel_finest"]),
            n_kernel_ladder=tuple(
                int(value) for value in payload["n_kernel_ladder"]
            ),
            n_de_points=int(payload["n_de_points"]),
            max_de_steps=int(payload["max_de_steps"]),
            root_maxfev=int(payload["root_maxfev"]),
            theta_b_seed_policy=str(payload["theta_b_seed_policy"]),
            theta_b_seed_axis_deg=tuple(
                float(value) for value in payload["theta_b_seed_axis_deg"]
            ),
            theta_b_chart_span_deg=float(payload["theta_b_chart_span_deg"]),
            theta_b_chart_count=int(payload["theta_b_chart_count"]),
            theta_b_chart_source_id=str(payload["theta_b_chart_source_id"]),
            kdf_seed_axis=tuple(
                float(value) for value in payload["kdf_seed_axis"]
            ),
            stop_at=str(payload["stop_at"]),
            scan_mode=str(payload["scan_mode"]),
            third_residual=str(payload["third_residual"]),
            residual_weights=tuple(
                float(value) for value in payload["residual_weights"]
            ),
            max_mach=float(payload["max_mach"]),
            closure_model=str(payload["closure_model"]),
            backend=str(payload.get("backend", "")),
            rt_scale_invariant=bool(payload.get("rt_scale_invariant", False)),
        )

    @property
    def identity_sha256(self) -> str:
        payload = self.to_dict()
        # The equations, normalized residuals, theta_B/kdf roots, and Eq. (5)
        # dR/R verdict are invariant under a uniform Rt scale.  Rt_reference is
        # retained as provenance but cannot make an otherwise identical query
        # miss the domain.
        payload.pop("Rt_reference")
        return _payload_sha256(payload)


@dataclass(frozen=True)
class RaoKernelResolutionEvidence:
    """Kernel-refinement evidence attached to one design-grid sample."""

    n_kernel: tuple[int, ...]
    root_success: tuple[bool, ...]
    theta_b_deg: tuple[float | None, ...]
    kdf: tuple[float | None, ...]
    residual_norm_geometry: tuple[float | None, ...]
    status: str
    reason: str
    theta_delta_tolerance_deg: float
    kdf_delta_tolerance: float

    def __post_init__(self) -> None:
        count = len(self.n_kernel)
        if not (
            len(self.root_success)
            == len(self.theta_b_deg)
            == len(self.kdf)
            == len(self.residual_norm_geometry)
            == count
        ):
            raise ValueError("kernel-resolution evidence arrays must align")
        if any(int(value) < 5 for value in self.n_kernel):
            raise ValueError("all n_kernel values must be >= 5")
        if tuple(sorted(set(self.n_kernel))) != self.n_kernel:
            raise ValueError("n_kernel values must be strictly increasing")
        if self.status not in RESOLUTION_STATUSES:
            raise ValueError(f"unknown resolution status {self.status!r}")
        if not self.reason:
            raise ValueError("resolution evidence requires a reason")
        for name in ("theta_delta_tolerance_deg", "kdf_delta_tolerance"):
            value = _require_finite(name, getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        for name, values in (
            ("theta_b_deg", self.theta_b_deg),
            ("kdf", self.kdf),
            ("residual_norm_geometry", self.residual_norm_geometry),
        ):
            for index, value in enumerate(values):
                if value is not None:
                    _require_finite(f"{name}[{index}]", value)
        if self.status == RESOLUTION_PASSED:
            if count < 2 or not all(self.root_success):
                raise ValueError(
                    "passed kernel resolution requires every rung to converge"
                )
            final_values = (
                self.theta_b_deg[-2],
                self.theta_b_deg[-1],
                self.kdf[-2],
                self.kdf[-1],
                self.residual_norm_geometry[-1],
            )
            if any(value is None for value in final_values):
                raise ValueError(
                    "passed kernel resolution requires finite final-rung evidence"
                )
            if (
                abs(self.theta_b_deg[-1] - self.theta_b_deg[-2])
                > self.theta_delta_tolerance_deg
                or abs(self.kdf[-1] - self.kdf[-2])
                > self.kdf_delta_tolerance
            ):
                raise ValueError(
                    "passed kernel resolution exceeds its declared final-rung tolerances"
                )

    @classmethod
    def not_evaluated(
        cls,
        *,
        theta_delta_tolerance_deg: float,
        kdf_delta_tolerance: float,
    ) -> "RaoKernelResolutionEvidence":
        return cls(
            n_kernel=(),
            root_success=(),
            theta_b_deg=(),
            kdf=(),
            residual_norm_geometry=(),
            status=RESOLUTION_NOT_EVALUATED,
            reason="kernel refinement was not evaluated",
            theta_delta_tolerance_deg=float(theta_delta_tolerance_deg),
            kdf_delta_tolerance=float(kdf_delta_tolerance),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kdf": list(self.kdf),
            "kdf_delta_tolerance": float(self.kdf_delta_tolerance),
            "n_kernel": [int(value) for value in self.n_kernel],
            "reason": self.reason,
            "residual_norm_geometry": list(self.residual_norm_geometry),
            "root_success": [bool(value) for value in self.root_success],
            "status": self.status,
            "theta_b_deg": list(self.theta_b_deg),
            "theta_delta_tolerance_deg": float(
                self.theta_delta_tolerance_deg
            ),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "RaoKernelResolutionEvidence":
        return cls(
            n_kernel=tuple(int(value) for value in payload["n_kernel"]),
            root_success=tuple(
                bool(value) for value in payload["root_success"]
            ),
            theta_b_deg=tuple(
                _optional_finite_float(value)
                for value in payload["theta_b_deg"]
            ),
            kdf=tuple(
                _optional_finite_float(value) for value in payload["kdf"]
            ),
            residual_norm_geometry=tuple(
                _optional_finite_float(value)
                for value in payload["residual_norm_geometry"]
            ),
            status=str(payload["status"]),
            reason=str(payload["reason"]),
            theta_delta_tolerance_deg=float(
                payload["theta_delta_tolerance_deg"]
            ),
            kdf_delta_tolerance=float(payload["kdf_delta_tolerance"]),
        )


@dataclass(frozen=True)
class RaoConvergenceSample:
    """One fixed-design convergence and Rao-validity classification."""

    epsilon: float
    length_pct: float
    classification: str
    reason: str
    theta_b_deg: float | None
    kdf: float | None
    residual_norm_geometry: float | None
    rao_eq5_evaluable: bool
    rao_eq5_valid: bool
    rao_eq5_min_dr_over_r: float | None
    resolution: RaoKernelResolutionEvidence
    source_scan_sha256: str

    def __post_init__(self) -> None:
        if _require_finite("epsilon", self.epsilon) <= 1.0:
            raise ValueError("sample epsilon must be > 1")
        _require_finite("length_pct", self.length_pct)
        if self.classification not in SAMPLE_CLASSIFICATIONS:
            raise ValueError(
                f"unknown sample classification {self.classification!r}"
            )
        if not self.reason:
            raise ValueError("sample classification requires a reason")
        for name in (
            "theta_b_deg",
            "kdf",
            "residual_norm_geometry",
            "rao_eq5_min_dr_over_r",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_finite(name, value)
        if len(self.source_scan_sha256) != 64:
            raise ValueError("source_scan_sha256 must be a SHA-256 hex digest")
        try:
            int(self.source_scan_sha256, 16)
        except ValueError as exc:
            raise ValueError(
                "source_scan_sha256 must be a SHA-256 hex digest"
            ) from exc
        if self.classification == SAMPLE_CONVERGED_VALID:
            if not (self.rao_eq5_evaluable and self.rao_eq5_valid):
                raise ValueError(
                    "converged_valid requires an evaluable, valid Rao Eq. (5) result"
                )
            if self.resolution.status != RESOLUTION_PASSED:
                raise ValueError(
                    "converged_valid requires passed kernel-resolution evidence"
                )
        if self.classification == SAMPLE_CONVERGED_INVALID:
            if not self.rao_eq5_evaluable or self.rao_eq5_valid:
                raise ValueError(
                    "converged_invalid requires an evaluable, invalid Rao Eq. (5) result"
                )
            if self.resolution.status != RESOLUTION_PASSED:
                raise ValueError(
                    "converged_invalid requires passed kernel-resolution evidence"
                )
        if (
            self.classification == SAMPLE_UNDER_RESOLVED
            and self.resolution.status == RESOLUTION_PASSED
        ):
            raise ValueError(
                "under_resolved requires failed or unevaluated kernel evidence"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "epsilon": float(self.epsilon),
            "kdf": self.kdf,
            "length_pct": float(self.length_pct),
            "rao_eq5_evaluable": bool(self.rao_eq5_evaluable),
            "rao_eq5_min_dr_over_r": self.rao_eq5_min_dr_over_r,
            "rao_eq5_valid": bool(self.rao_eq5_valid),
            "reason": self.reason,
            "residual_norm_geometry": self.residual_norm_geometry,
            "resolution": self.resolution.to_dict(),
            "source_scan_sha256": self.source_scan_sha256,
            "theta_b_deg": self.theta_b_deg,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RaoConvergenceSample":
        return cls(
            epsilon=float(payload["epsilon"]),
            length_pct=float(payload["length_pct"]),
            classification=str(payload["classification"]),
            reason=str(payload["reason"]),
            theta_b_deg=_optional_finite_float(payload.get("theta_b_deg")),
            kdf=_optional_finite_float(payload.get("kdf")),
            residual_norm_geometry=_optional_finite_float(
                payload.get("residual_norm_geometry")
            ),
            rao_eq5_evaluable=bool(payload["rao_eq5_evaluable"]),
            rao_eq5_valid=bool(payload["rao_eq5_valid"]),
            rao_eq5_min_dr_over_r=_optional_finite_float(
                payload.get("rao_eq5_min_dr_over_r")
            ),
            resolution=RaoKernelResolutionEvidence.from_dict(
                payload["resolution"]
            ),
            source_scan_sha256=str(payload["source_scan_sha256"]),
        )


@dataclass(frozen=True)
class RaoDomainLookup:
    """Fail-closed result of one declared-domain lookup."""

    status: str
    reason: str
    epsilon: float
    length_pct: float
    epsilon_bracket: tuple[float, float] | None
    length_pct_bracket: tuple[float, float] | None
    sample_classifications: tuple[str, ...]
    domain_sha256: str
    source_model_id: str
    context_sha256: str

    def __post_init__(self) -> None:
        if self.status not in DOMAIN_LOOKUP_STATUSES:
            raise ValueError(f"unknown lookup status {self.status!r}")
        if not self.reason:
            raise ValueError("domain lookup requires a reason")

    @property
    def supported(self) -> bool:
        return self.status == DOMAIN_SUPPORTED

    def require_supported(self) -> "RaoDomainLookup":
        if not self.supported:
            raise RaoConvergenceDomainError(self)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_sha256": self.context_sha256,
            "domain_sha256": self.domain_sha256,
            "epsilon": _optional_finite_float(self.epsilon),
            "epsilon_bracket": (
                list(self.epsilon_bracket)
                if self.epsilon_bracket is not None
                else None
            ),
            "length_pct": _optional_finite_float(self.length_pct),
            "length_pct_bracket": (
                list(self.length_pct_bracket)
                if self.length_pct_bracket is not None
                else None
            ),
            "reason": self.reason,
            "sample_classifications": list(self.sample_classifications),
            "source_model_id": self.source_model_id,
            "status": self.status,
            "supported": self.supported,
        }


class RaoConvergenceDomainError(RuntimeError):
    """Typed refusal for a query outside a certified convergence cell."""

    def __init__(self, lookup: RaoDomainLookup):
        self.lookup = lookup
        super().__init__(
            f"Rao convergence-domain refusal [{lookup.status}]: "
            f"{lookup.reason}"
        )


def _strict_axis(name: str, values: Sequence[float]) -> tuple[float, ...]:
    axis = tuple(
        _require_finite(f"{name}[{i}]", value)
        for i, value in enumerate(values)
    )
    if len(axis) < 2:
        raise ValueError(f"{name} must contain at least two points")
    if any(right <= left for left, right in zip(axis[:-1], axis[1:])):
        raise ValueError(f"{name} must be strictly increasing with no duplicates")
    return axis


def _axis_bracket(axis: tuple[float, ...], value: float) -> tuple[int, int] | None:
    if value < axis[0] or value > axis[-1]:
        return None
    position = bisect_left(axis, value)
    if position < len(axis) and axis[position] == value:
        return position, position
    if position == 0 or position == len(axis):
        return None
    return position - 1, position


@dataclass(frozen=True)
class RaoConvergenceDomain:
    """Versioned ``epsilon x length_pct`` provider-support declaration.

    This object is a conservative classifier, not a proof that a physical
    optimum does or does not exist between samples.  Numerical failures remain
    unknown.  An interpolated query is supported only when every bracketing
    sample is kernel-resolved, residual-converged, and Rao-Eq.-(5) valid.
    """

    context: RaoConvergenceContext
    epsilon_axis: tuple[float, ...]
    length_pct_axis: tuple[float, ...]
    samples: tuple[tuple[RaoConvergenceSample, ...], ...]
    residual_norm_limit: float
    theta_delta_tolerance_deg: float
    kdf_delta_tolerance: float
    max_epsilon_spacing: float
    max_length_pct_spacing: float
    schema_version: int = CONVERGENCE_DOMAIN_SCHEMA_VERSION
    source_model_id: str = CONVERGENCE_DOMAIN_SOURCE_MODEL_ID
    validity_source_id: str = field(default_factory=_rao_eq5_source_id)

    def __post_init__(self) -> None:
        epsilon_axis = _strict_axis("epsilon_axis", self.epsilon_axis)
        length_axis = _strict_axis("length_pct_axis", self.length_pct_axis)
        object.__setattr__(self, "epsilon_axis", epsilon_axis)
        object.__setattr__(self, "length_pct_axis", length_axis)
        if self.schema_version != CONVERGENCE_DOMAIN_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported convergence-domain schema {self.schema_version}"
            )
        if self.source_model_id != CONVERGENCE_DOMAIN_SOURCE_MODEL_ID:
            raise ValueError(
                f"unsupported convergence-domain source {self.source_model_id!r}"
            )
        if self.validity_source_id != _rao_eq5_source_id():
            raise ValueError(
                "convergence-domain validity source does not match the "
                "canonical Rao Eq. (5) implementation authority"
            )
        for name in (
            "residual_norm_limit",
            "theta_delta_tolerance_deg",
            "kdf_delta_tolerance",
            "max_epsilon_spacing",
            "max_length_pct_spacing",
        ):
            value = _require_finite(name, getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if len(self.samples) != len(epsilon_axis):
            raise ValueError("sample rows must match epsilon_axis")
        for i, row in enumerate(self.samples):
            if len(row) != len(length_axis):
                raise ValueError("sample columns must match length_pct_axis")
            for j, sample in enumerate(row):
                if sample.epsilon != epsilon_axis[i]:
                    raise ValueError("sample epsilon does not match its axis")
                if sample.length_pct != length_axis[j]:
                    raise ValueError("sample length_pct does not match its axis")
                if (
                    sample.classification != SAMPLE_NUMERICAL_FAILURE
                    and sample.resolution.n_kernel
                    != self.context.n_kernel_ladder
                ):
                    raise ValueError(
                        "sample kernel ladder does not match the domain context"
                    )
                if (
                    sample.resolution.theta_delta_tolerance_deg
                    != self.theta_delta_tolerance_deg
                    or sample.resolution.kdf_delta_tolerance
                    != self.kdf_delta_tolerance
                ):
                    raise ValueError(
                        "sample kernel tolerances do not match the domain policy"
                    )
                if sample.classification in {
                    SAMPLE_CONVERGED_VALID,
                    SAMPLE_CONVERGED_INVALID,
                }:
                    if sample.resolution.status != RESOLUTION_PASSED:
                        raise ValueError(
                            "converged samples require passed kernel-resolution evidence"
                        )
                    if sample.theta_b_deg is None or sample.kdf is None:
                        raise ValueError(
                            "converged samples require finite theta_B and kdf"
                        )
                    if (
                        sample.residual_norm_geometry is None
                        or sample.residual_norm_geometry
                        > self.residual_norm_limit
                    ):
                        raise ValueError(
                            "converged sample exceeds the domain residual limit"
                        )
                    if not sample.rao_eq5_evaluable:
                        raise ValueError(
                            "converged sample requires an evaluable Rao Eq. (5) verdict"
                        )
                    if (
                        sample.classification == SAMPLE_CONVERGED_VALID
                    ) != sample.rao_eq5_valid:
                        raise ValueError(
                            "converged sample classification disagrees with Rao Eq. (5)"
                        )
                if (
                    sample.classification == SAMPLE_UNDER_RESOLVED
                    and sample.resolution.status == RESOLUTION_PASSED
                ):
                    raise ValueError(
                        "under-resolved samples cannot carry passed resolution evidence"
                    )

    @property
    def epsilon_bounds(self) -> tuple[float, float]:
        return self.epsilon_axis[0], self.epsilon_axis[-1]

    @property
    def length_pct_bounds(self) -> tuple[float, float]:
        return self.length_pct_axis[0], self.length_pct_axis[-1]

    def classification_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in SAMPLE_CLASSIFICATIONS}
        for row in self.samples:
            for sample in row:
                counts[sample.classification] += 1
        return counts

    def _payload(self) -> dict[str, Any]:
        return {
            "axes": {
                "epsilon": list(self.epsilon_axis),
                "length_pct": list(self.length_pct_axis),
            },
            "bounds": {
                "epsilon": list(self.epsilon_bounds),
                "length_pct": list(self.length_pct_bounds),
            },
            "classification_counts": self.classification_counts(),
            "context": self.context.to_dict(),
            "policies": {
                "max_epsilon_spacing": float(self.max_epsilon_spacing),
                "max_length_pct_spacing": float(
                    self.max_length_pct_spacing
                ),
                "kdf_delta_tolerance": float(self.kdf_delta_tolerance),
                "residual_norm_limit": float(self.residual_norm_limit),
                "theta_delta_tolerance_deg": float(
                    self.theta_delta_tolerance_deg
                ),
            },
            "samples": [
                [sample.to_dict() for sample in row] for row in self.samples
            ],
            "schema_version": int(self.schema_version),
            "source_model_id": self.source_model_id,
            "validity_source_id": self.validity_source_id,
        }

    @property
    def artifact_sha256(self) -> str:
        return _payload_sha256(self._payload())

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["artifact_sha256"] = self.artifact_sha256
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
        ) + ("\n" if indent is not None else "")

    def write_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RaoConvergenceDomain":
        expected_hash = str(payload.get("artifact_sha256", ""))
        if len(expected_hash) != 64:
            raise ValueError("convergence-domain artifact SHA-256 is missing")
        unhashed = {
            key: value
            for key, value in payload.items()
            if key != "artifact_sha256"
        }
        actual_hash = _payload_sha256(unhashed)
        if actual_hash != expected_hash:
            raise ValueError(
                "convergence-domain artifact SHA-256 mismatch: "
                f"expected {expected_hash}, computed {actual_hash}"
            )
        axes = payload["axes"]
        policies = payload["policies"]
        domain = cls(
            context=RaoConvergenceContext.from_dict(payload["context"]),
            epsilon_axis=tuple(float(value) for value in axes["epsilon"]),
            length_pct_axis=tuple(
                float(value) for value in axes["length_pct"]
            ),
            samples=tuple(
                tuple(RaoConvergenceSample.from_dict(sample) for sample in row)
                for row in payload["samples"]
            ),
            residual_norm_limit=float(policies["residual_norm_limit"]),
            theta_delta_tolerance_deg=float(
                policies["theta_delta_tolerance_deg"]
            ),
            kdf_delta_tolerance=float(policies["kdf_delta_tolerance"]),
            max_epsilon_spacing=float(policies["max_epsilon_spacing"]),
            max_length_pct_spacing=float(
                policies["max_length_pct_spacing"]
            ),
            schema_version=int(payload["schema_version"]),
            source_model_id=str(payload["source_model_id"]),
            validity_source_id=str(payload["validity_source_id"]),
        )
        if domain.artifact_sha256 != expected_hash:
            raise ValueError("convergence-domain canonical reconstruction mismatch")
        return domain

    @classmethod
    def read_json(cls, path: str | Path) -> "RaoConvergenceDomain":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("convergence-domain JSON root must be an object")
        return cls.from_dict(payload)

    def _lookup_result(
        self,
        status: str,
        reason: str,
        epsilon: float,
        length_pct: float,
        epsilon_bracket: tuple[float, float] | None,
        length_pct_bracket: tuple[float, float] | None,
        classifications: Iterable[str] = (),
    ) -> RaoDomainLookup:
        return RaoDomainLookup(
            status=status,
            reason=reason,
            epsilon=float(epsilon),
            length_pct=float(length_pct),
            epsilon_bracket=epsilon_bracket,
            length_pct_bracket=length_pct_bracket,
            sample_classifications=tuple(classifications),
            domain_sha256=self.artifact_sha256,
            source_model_id=self.source_model_id,
            context_sha256=self.context.identity_sha256,
        )

    def lookup(
        self,
        epsilon: float,
        length_pct: float,
        *,
        expected_context: RaoConvergenceContext,
    ) -> RaoDomainLookup:
        """Classify a provider request without extrapolation or fallback."""
        epsilon = float(epsilon)
        length_pct = float(length_pct)
        if expected_context.closure_model != self.context.closure_model:
            return self._lookup_result(
                DOMAIN_MODEL_MISMATCH,
                "requested closure model does not match the domain artifact",
                epsilon,
                length_pct,
                None,
                None,
            )
        if expected_context.identity_sha256 != self.context.identity_sha256:
            return self._lookup_result(
                DOMAIN_CONTEXT_MISMATCH,
                "requested fixed design/mesh context does not match the domain artifact",
                epsilon,
                length_pct,
                None,
                None,
            )
        if not math.isfinite(epsilon) or not math.isfinite(length_pct):
            return self._lookup_result(
                DOMAIN_UNKNOWN_OUTSIDE,
                "non-finite design coordinates are outside the declared domain",
                epsilon,
                length_pct,
                None,
                None,
            )
        epsilon_indices = _axis_bracket(self.epsilon_axis, epsilon)
        length_indices = _axis_bracket(self.length_pct_axis, length_pct)
        if epsilon_indices is None or length_indices is None:
            return self._lookup_result(
                DOMAIN_UNKNOWN_OUTSIDE,
                "design coordinates are outside the declared epsilon/length bounds",
                epsilon,
                length_pct,
                None,
                None,
            )
        ei0, ei1 = epsilon_indices
        li0, li1 = length_indices
        epsilon_bracket = (self.epsilon_axis[ei0], self.epsilon_axis[ei1])
        length_bracket = (
            self.length_pct_axis[li0],
            self.length_pct_axis[li1],
        )
        if (
            self.epsilon_axis[ei1] - self.epsilon_axis[ei0]
            > self.max_epsilon_spacing
            or self.length_pct_axis[li1] - self.length_pct_axis[li0]
            > self.max_length_pct_spacing
        ):
            return self._lookup_result(
                DOMAIN_UNKNOWN_UNDER_RESOLVED,
                "local design-grid spacing exceeds the declared resolution policy",
                epsilon,
                length_pct,
                epsilon_bracket,
                length_bracket,
            )
        corners = tuple(
            self.samples[i][j]
            for i in sorted(set(epsilon_indices))
            for j in sorted(set(length_indices))
        )
        classifications = tuple(sample.classification for sample in corners)
        classification_set = set(classifications)
        if SAMPLE_UNDER_RESOLVED in classification_set:
            status = DOMAIN_UNKNOWN_UNDER_RESOLVED
            reason = "one or more bracketing samples lack converged mesh evidence"
        elif classification_set.intersection(
            {SAMPLE_NUMERICAL_FAILURE, SAMPLE_UNEVALUABLE}
        ):
            status = DOMAIN_UNKNOWN_SCAN_FAILURE
            reason = (
                "one or more bracketing samples are numerically failed or unevaluable; "
                "this is not evidence of physical non-existence"
            )
        elif classification_set == {SAMPLE_CONVERGED_VALID}:
            status = DOMAIN_SUPPORTED
            reason = (
                "all bracketing samples are residual-converged, kernel-resolved, "
                "and Rao Eq. (5) valid"
            )
        elif classification_set == {SAMPLE_CONVERGED_INVALID}:
            status = DOMAIN_REFUSED_INVALID_RAO_REGION
            reason = "all bracketing samples are below the Rao Eq. (5) boundary"
        elif classification_set == {SAMPLE_NOT_CONVERGED}:
            status = DOMAIN_REFUSED_NOT_CONVERGED
            reason = (
                "all bracketing samples exceed the declared closure-residual limit"
            )
        else:
            status = DOMAIN_UNKNOWN_UNDER_RESOLVED
            reason = (
                "bracketing classifications disagree; the local boundary is "
                "under-resolved"
            )
        return self._lookup_result(
            status,
            reason,
            epsilon,
            length_pct,
            epsilon_bracket,
            length_bracket,
            classifications,
        )

    def require_supported(
        self,
        epsilon: float,
        length_pct: float,
        *,
        expected_context: RaoConvergenceContext,
    ) -> RaoDomainLookup:
        return self.lookup(
            epsilon, length_pct, expected_context=expected_context
        ).require_supported()


def target_length(Rt: float, epsilon: float, length_pct: float) -> float:
    """Rao/TOP reduced bell length convention used elsewhere in the repo."""
    Re = math.sqrt(epsilon) * Rt
    return (length_pct / 100.0) * ((Re - Rt) / math.tan(math.radians(15.0)))


def ideal_exit_cf(config: ExistenceScanConfig) -> float:
    """Ideal quasi-1D thrust coefficient target for the requested exit area."""
    Me = mach_from_area_ratio(config.epsilon, config.gamma, supersonic=True)
    Pe = isentropic_pressure_ratio(Me, config.gamma)
    return thrust_coefficient(
        Me, config.gamma, Pe, config.pa_over_p0, config.epsilon,
    )


def _moc_node_from_flow(p, gamma: float) -> MOCNode:
    return MOCNode(
        float(p.x), float(p.r), max(float(p.M), 1.000001),
        float(p.theta), gamma,
    )


def prandtl_meyer_fan_post_state(
    pre: MOCNode,
    turn_angle: float,
    gamma: float,
    *,
    max_mach: float = 12.0,
) -> MOCNode:
    """Return the post-fan D state for ``dnu = dtheta = turn_angle``.

    Positive ``turn_angle`` is an expansion fan under the sign convention in
    the current Rao kernel.  Negative values are mathematically evaluable until
    ``nu_post`` would become negative, but they represent compression and are
    not part of the default fan scan.
    """
    M_pre = max(float(pre.M), 1.000001)
    nu_pre = prandtl_meyer(M_pre, gamma)
    nu_post = nu_pre + float(turn_angle)
    if nu_post < 0.0:
        raise ValueError("fan turn would require negative Prandtl-Meyer angle")
    M_post = mach_from_prandtl_meyer(nu_post, gamma)
    if not math.isfinite(M_post) or M_post < 1.0:
        raise ValueError("fan produced an invalid post-fan Mach number")
    if M_post > max_mach:
        raise ValueError("fan post-fan Mach exceeds max_mach")
    return MOCNode(
        x=float(pre.x),
        r=float(pre.r),
        M=float(max(M_post, 1.000001)),
        theta=float(pre.theta + turn_angle),
        gamma=float(gamma),
    )


def _annular_mdot(p_lo: MOCNode, p_hi: MOCNode) -> float:
    """Mass contribution over one DE segment, matching ``nasa_moc``."""
    dr = p_hi.r - p_lo.r
    if abs(dr) <= 1e-15:
        return 0.0
    dxdr = (p_hi.x - p_lo.x) / dr
    rho_u_avg = 0.5 * (
        p_lo.rho * p_lo.u + p_hi.rho * p_hi.u
        - dxdr * (p_lo.rho * p_lo.v + p_hi.rho * p_hi.v)
    )
    da = math.pi * (p_hi.r * p_hi.r - p_lo.r * p_lo.r)
    return float(rho_u_avg * da)


def _integrate_to_radius(
    D: MOCNode,
    target_radius_value: float,
    gamma: float,
    *,
    n_steps: int,
    max_steps: int,
) -> DEIntegrationResult:
    if D.r >= target_radius_value:
        return DEIntegrationResult((D,), 0.0, False, 4)
    nodes = [D]
    p0 = D
    mass_total = 0.0
    nominal_h = (target_radius_value - D.r) / max(int(n_steps), 1)
    for _ in range(max_steps):
        h = min(max(nominal_h, 1e-9), target_radius_value - p0.r)
        if h <= 1e-12:
            return DEIntegrationResult(tuple(nodes), mass_total, True, 0)
        rk = None
        for _attempt in range(16):
            rk = nasa_runge_kutta(h, p0.r, p0.x, p0.M, p0.theta, gamma)
            if rk is not None and all(math.isfinite(v) for v in rk) and rk[3] > p0.r:
                break
            h *= 0.5
            rk = None
        if rk is None:
            return DEIntegrationResult(tuple(nodes), mass_total, False, 4)
        p1 = MOCNode(float(rk[1]), float(rk[3]), max(float(rk[0]), 1.000001),
                     float(rk[2]), gamma)
        mass_total += _annular_mdot(p0, p1)
        nodes.append(p1)
        p0 = p1
        if p0.r >= target_radius_value - 1e-10:
            return DEIntegrationResult(tuple(nodes), mass_total, True, 0)
    return DEIntegrationResult(tuple(nodes), mass_total, False, 4)


def _integrate_to_length(
    D: MOCNode,
    target_length_value: float,
    gamma: float,
    *,
    target_radius_value: float,
    n_steps: int,
    max_steps: int,
) -> DEIntegrationResult:
    if D.x >= target_length_value:
        return DEIntegrationResult((D,), 0.0, False, 4)
    nodes = [D]
    p0 = D
    mass_total = 0.0
    h_base = max((target_radius_value - D.r) / max(int(n_steps), 1), 1e-5)
    for _ in range(max_steps):
        rk = None
        h = h_base
        for _attempt in range(16):
            rk = nasa_runge_kutta(h, p0.r, p0.x, p0.M, p0.theta, gamma)
            if rk is not None and all(math.isfinite(v) for v in rk) and rk[3] > p0.r:
                break
            h *= 0.5
            rk = None
        if rk is None:
            return DEIntegrationResult(tuple(nodes), mass_total, False, 4)

        p_try = MOCNode(float(rk[1]), float(rk[3]), max(float(rk[0]), 1.000001),
                        float(rk[2]), gamma)
        if p_try.x >= target_length_value:
            h_lo, h_hi = 0.0, h
            best = p_try
            for _bisect in range(40):
                h_mid = 0.5 * (h_lo + h_hi)
                rk_mid = nasa_runge_kutta(
                    h_mid, p0.r, p0.x, p0.M, p0.theta, gamma,
                )
                if rk_mid is None or not all(math.isfinite(v) for v in rk_mid):
                    h_hi = h_mid
                    continue
                p_mid = MOCNode(
                    float(rk_mid[1]), float(rk_mid[3]),
                    max(float(rk_mid[0]), 1.000001), float(rk_mid[2]), gamma,
                )
                if p_mid.x < target_length_value:
                    h_lo = h_mid
                else:
                    h_hi = h_mid
                    best = p_mid
                if abs(best.x - target_length_value) / max(target_length_value, 1e-12) < 1e-10:
                    break
            mass_total += _annular_mdot(p0, best)
            nodes.append(best)
            return DEIntegrationResult(tuple(nodes), mass_total, True, 0)

        mass_total += _annular_mdot(p0, p_try)
        nodes.append(p_try)
        p0 = p_try
        if p0.r > 1.5 * target_radius_value:
            return DEIntegrationResult(tuple(nodes), mass_total, False, 4)
    return DEIntegrationResult(tuple(nodes), mass_total, False, 4)


def integrate_de(
    D: MOCNode,
    mass_target: float,
    config: ExistenceScanConfig,
) -> DEIntegrationResult:
    """Integrate the downstream DE characteristic from a supplied D state."""
    if config.stop_at == STOP_MASS:
        nodes, mass = find_point_e(
            D, mass_target, config.gamma,
            n_steps=config.n_de_points,
            max_steps=config.max_de_steps,
        )
        ok = bool(nodes) and math.isfinite(mass) and len(nodes) >= 2
        return DEIntegrationResult(tuple(nodes), float(mass), ok, 0 if ok else 4)
    if config.stop_at == STOP_RADIUS:
        return _integrate_to_radius(
            D, config.target_radius, config.gamma,
            n_steps=config.n_de_points, max_steps=config.max_de_steps,
        )
    if config.stop_at == STOP_LENGTH:
        return _integrate_to_length(
            D, config.target_length, config.gamma,
            target_radius_value=config.target_radius,
            n_steps=config.n_de_points, max_steps=config.max_de_steps,
        )
    raise ValueError(f"stop_at must be one of {STOP_MODES}, got {config.stop_at!r}")


def _rao_theta_calc(state: MOCNode, gamma: float, pa_over_p0: float) -> float:
    """Rao free-exit transversality angle at E.

    This is the NASA/Rao free-endpoint condition.  For the fixed endpoint
    problem in this repo (fixed exit radius and fixed length), the natural
    endpoint transversality condition is not a hard boundary condition.  We
    therefore report ``sigma_E = theta_E - theta_Rao(E)`` separately from
    geometry closure and from the BVP's interior algebraic stationarity.
    """
    if state.M <= 1.0:
        return float("nan")
    denom = state.rho * state.V * state.V * math.tan(state.mu)
    if denom <= 0.0:
        return float("nan")
    arg = 2.0 * (state.p - pa_over_p0) / denom
    if not (-1.0 <= arg <= 1.0):
        return float("nan")
    return 0.5 * math.asin(arg)


def sigma_E_rad(state: MOCNode, gamma: float, pa_over_p0: float) -> tuple[float, float]:
    """Return ``(sigma_E, theta_Rao_E)`` in radians.

    ``sigma_E`` is absolute radians, not the older relative
    ``rao_exit_residual`` ratio.  Non-finite ``theta_Rao_E`` yields
    ``(nan, nan)`` so stationarity-mode scans cannot hide the failure.
    """
    theta_calc = _rao_theta_calc(state, gamma, pa_over_p0)
    if not math.isfinite(theta_calc):
        return float("nan"), float("nan")
    return float(state.theta - theta_calc), float(theta_calc)


def stationarity_derivation_summary() -> str:
    """Short canonical note used by scripts and docs.

    The key distinction is fixed-endpoint vs free-endpoint.  The scan's
    ``sigma_E`` is the free-exit transversality diagnostic.  The fixed
    (radius,length) BVP's hard stationarity is the interior algebraic Rao
    condition plus fixed endpoint constraints; the free endpoint condition is
    not automatically required.
    """
    return (
        "For fixed exit radius and fixed nozzle length, theta_E - theta_Rao(E) "
        "is a free-endpoint transversality diagnostic, not automatically the "
        "hard stationarity condition.  Geometry mode closes radius/length; "
        "stationarity mode adds absolute sigma_E in radians to test whether "
        "the fixed endpoint root also satisfies the free-exit condition."
    )


def _post_state_candidates(
    pre: MOCNode,
    model: str,
    config: ExistenceScanConfig,
) -> Iterable[tuple[MOCNode | None, float, int]]:
    """Yield ``(post_node, fan_turn_deg, failure_code)`` candidates."""
    if model == MODEL_SMOOTH:
        yield MOCNode(pre.x, pre.r, max(pre.M, 1.000001), pre.theta, config.gamma), 0.0, 0
        return

    if model == MODEL_POSITION:
        th_offsets = np.linspace(
            -float(config.position_theta_span_deg),
            float(config.position_theta_span_deg),
            max(int(config.position_theta_count), 1),
        )
        M_values = np.linspace(
            max(1.000001, float(pre.M) - float(config.position_mach_down)),
            min(float(config.max_mach), float(pre.M) + float(config.position_mach_up)),
            max(int(config.position_mach_count), 1),
        )
        for dth_deg in th_offsets:
            theta = float(pre.theta + math.radians(float(dth_deg)))
            for M in M_values:
                if M <= 1.0 or not math.isfinite(M):
                    yield None, 0.0, 3
                    continue
                yield MOCNode(pre.x, pre.r, float(M), theta, config.gamma), 0.0, 0
        return

    if model == MODEL_FAN:
        turn_values = np.linspace(
            float(config.fan_turn_min_deg),
            float(config.fan_turn_max_deg),
            max(int(config.fan_turn_count), 1),
        )
        for turn_deg in turn_values:
            try:
                post = prandtl_meyer_fan_post_state(
                    pre, math.radians(float(turn_deg)), config.gamma,
                    max_mach=config.max_mach,
                )
            except ValueError:
                yield None, float(turn_deg), 3
                continue
            yield post, float(turn_deg), 0
        return

    raise ValueError(f"unknown closure model {model!r}")


def _third_component_name(config: ExistenceScanConfig) -> str:
    if config.third_residual == THIRD_AUTO:
        return THIRD_PERFORMANCE if config.stop_at == STOP_MASS else THIRD_MASS
    if config.third_residual not in THIRD_COMPONENTS:
        raise ValueError(
            f"third_residual must be one of {THIRD_COMPONENTS}, "
            f"got {config.third_residual!r}"
        )
    return config.third_residual


def _evaluate_candidate(
    pre_D: MOCNode,
    post_D: MOCNode,
    mass_BD: float,
    config: ExistenceScanConfig,
    cf_target: float,
) -> dict:
    result = integrate_de(post_D, mass_BD, config)
    if not result.success or result.E is None:
        return {"success": False, "failure_code": result.failure_code}

    # B4's Eq. (5) verdict is evaluated on the same DE nodes that produced the
    # scan cell.  Import locally so this NumPy scan module stays independent of
    # the variational solver during module import.
    from raosim.rao_variational import rao_radial_increment_validity

    rao_validity = rao_radial_increment_validity(result.nodes)

    E = result.E
    radius_res = (float(E.r) - config.target_radius) / max(config.target_radius, 1e-12)
    length_res = (float(E.x) - config.target_length) / max(config.target_length, 1e-12)
    mass_ref = max(abs(mass_BD), abs(result.mass), 1e-12)
    mass_res = (float(result.mass) - float(mass_BD)) / mass_ref
    try:
        cf = surface_thrust_coefficient(
            result.nodes, config.gamma, config.Rt, config.pa_over_p0,
        )
        perf_res = (cf - cf_target) / max(abs(cf_target), 1e-12)
    except Exception:
        cf = float("nan")
        perf_res = float("nan")

    sigma, theta_calc = sigma_E_rad(E, config.gamma, config.pa_over_p0)
    if math.isfinite(theta_calc) and abs(theta_calc) > 1e-12:
        rao_exit_res = sigma / abs(theta_calc)
    else:
        rao_exit_res = float("nan")

    if config.scan_mode == SCAN_GEOMETRY:
        weights = np.asarray(config.residual_weights[:2], dtype=float)
        components = np.asarray([radius_res, length_res], dtype=float)
    elif config.scan_mode == SCAN_STATIONARITY:
        weights = np.asarray(config.residual_weights, dtype=float)
        components = np.asarray([radius_res, length_res, sigma], dtype=float)
    elif config.scan_mode == SCAN_DIAGNOSTIC:
        # Diagnostics are reported below; the norm remains a geometry norm.
        weights = np.asarray(config.residual_weights[:2], dtype=float)
        components = np.asarray([radius_res, length_res], dtype=float)
    else:
        raise ValueError(f"scan_mode must be one of {SCAN_MODES}, got {config.scan_mode!r}")

    if not np.all(np.isfinite(components)):
        norm = float("inf")
        code = 5
        success = False
    else:
        norm = float(np.sqrt(np.mean((weights * components) ** 2)))
        code = 0
        success = True

    return {
        "success": success,
        "failure_code": code,
        "residual_norm": norm,
        "radius_residual": float(radius_res),
        "length_residual": float(length_res),
        "sigma_E_rad": float(sigma),
        "theta_rao_E_deg": float(math.degrees(theta_calc)) if math.isfinite(theta_calc) else float("nan"),
        "mass_residual": float(mass_res),
        "performance_residual": float(perf_res),
        "rao_exit_residual": float(rao_exit_res),
        "thrust_coefficient": float(cf),
        "thrust_coefficient_target": float(cf_target),
        "d_mach_pre": float(pre_D.M),
        "d_mach_post": float(post_D.M),
        "d_mach_jump": float(post_D.M - pre_D.M),
        "d_x": float(pre_D.x),
        "d_r": float(pre_D.r),
        "d_theta_pre_deg": float(math.degrees(pre_D.theta)),
        "d_theta_post_deg": float(math.degrees(post_D.theta)),
        "d_theta_jump_deg": float(math.degrees(post_D.theta - pre_D.theta)),
        "exit_x": float(E.x),
        "exit_r": float(E.r),
        "exit_angle_deg": float(math.degrees(E.theta)),
        "exit_mach": float(E.M),
        "rao_eq5_evaluable": bool(rao_validity.available),
        "rao_eq5_valid": bool(rao_validity.valid),
        "rao_eq5_min_dr_over_r": (
            float(rao_validity.minimum_predicted_dR_over_R)
            if rao_validity.minimum_predicted_dR_over_R is not None
            else float("nan")
        ),
    }


def evaluate_closure_point(
    config: ExistenceScanConfig,
    *,
    model: str,
    theta_B_deg: float,
    kdf: float,
    fan_turn_deg: float = 0.0,
) -> dict:
    """Evaluate one deterministic closure point.

    This is the scalar version of one scan cell.  It is used by root
    refinement and solver comparison so they do not have to reverse-engineer
    heatmap arrays.
    """
    if model == MODEL_POSITION:
        raise ValueError(
            "evaluate_closure_point does not support position-only roots; "
            "that model needs explicit post-D theta/M degrees of freedom."
        )
    kernel = build_kernel(
        config.Rt,
        config.throat_downstream_radius_factor * config.Rt,
        math.radians(float(theta_B_deg)),
        config.gamma,
        config.n_kernel,
        starting_line_method=config.starting_line_method,
        Ru=config.throat_upstream_radius_factor * config.Rt,
    )
    kernel_bd = tuple(node.to_flow_node() for node in kernel.bd)
    mass_BD, bd_segment = calc_mdot_bd(kernel_bd, float(kdf), config.gamma)
    if mass_BD <= 0.0 or not bd_segment:
        raise ValueError("nonpositive BD mass or empty BD segment")
    pre_D = _moc_node_from_flow(bd_segment[-1], config.gamma)
    if model == MODEL_SMOOTH:
        post_D = MOCNode(pre_D.x, pre_D.r, pre_D.M, pre_D.theta, config.gamma)
        turn = 0.0
    elif model == MODEL_FAN:
        turn = float(fan_turn_deg)
        post_D = prandtl_meyer_fan_post_state(
            pre_D, math.radians(turn), config.gamma, max_mach=config.max_mach,
        )
    else:
        raise ValueError(f"unknown closure model {model!r}")
    out = _evaluate_candidate(pre_D, post_D, mass_BD, config, ideal_exit_cf(config))
    out.update({
        "theta_B_deg": float(theta_B_deg),
        "kdf": float(kdf),
        "fan_turn_deg": float(turn),
        "n_kernel": int(config.n_kernel),
    })
    return out


def _root_result_from_eval(
    model: str,
    message: str,
    success: bool,
    ev: dict,
) -> RootRefineResult:
    geom = math.sqrt(
        0.5 * (float(ev["radius_residual"]) ** 2 + float(ev["length_residual"]) ** 2)
    )
    stat_components = np.asarray([
        float(ev["radius_residual"]),
        float(ev["length_residual"]),
        float(ev["sigma_E_rad"]),
    ], dtype=float)
    if np.all(np.isfinite(stat_components)):
        stat = float(np.sqrt(np.mean(stat_components ** 2)))
    else:
        stat = float("inf")
    return RootRefineResult(
        model=model,
        success=bool(success),
        message=str(message),
        n_kernel=int(ev.get("n_kernel", -1)),
        theta_B_deg=float(ev["theta_B_deg"]),
        kdf=float(ev["kdf"]),
        fan_turn_deg=float(ev.get("fan_turn_deg", 0.0)),
        radius_residual=float(ev["radius_residual"]),
        length_residual=float(ev["length_residual"]),
        sigma_E_rad=float(ev["sigma_E_rad"]),
        mass_residual=float(ev["mass_residual"]),
        performance_residual=float(ev["performance_residual"]),
        rao_exit_residual=float(ev["rao_exit_residual"]),
        residual_norm_geometry=geom,
        residual_norm_stationarity=stat,
        D_M_pre=float(ev["d_mach_pre"]),
        D_M_post=float(ev["d_mach_post"]),
        D_theta_pre_deg=float(ev["d_theta_pre_deg"]),
        D_theta_post_deg=float(ev["d_theta_post_deg"]),
        exit_angle_deg=float(ev["exit_angle_deg"]),
        exit_mach=float(ev["exit_mach"]),
    )


def refine_geometry_root(
    config: ExistenceScanConfig,
    *,
    model: str,
    theta_B_seed_deg: float,
    kdf_seed: float,
    fan_turn_deg: float = 0.0,
    maxfev: int = 80,
) -> RootRefineResult:
    """Refine ``(theta_B, kdf)`` so radius and length close.

    The hard residuals are only ``[(r_E-r_target)/r_target,
    (x_E-L_target)/L_target]``.  ``sigma_E`` is evaluated after the geometry
    root is found.  This matches the fixed-endpoint question: first ask
    whether smooth geometry closes, then ask whether that geometry root also
    satisfies the free-endpoint transversality diagnostic.
    """
    if model not in (MODEL_SMOOTH, MODEL_FAN):
        raise ValueError("root refinement currently supports smooth and fan models")
    root_config = replace(
        config,
        stop_at=STOP_MASS,
        scan_mode=SCAN_GEOMETRY,
        models=(model,),
        theta_b_values_deg=None,
        kdf_values=None,
    )

    def residual(z):
        theta_deg = float(z[0])
        kdf = float(z[1])
        if kdf <= 0.0 or kdf >= 1.0 or not math.isfinite(theta_deg):
            return np.array([1e3 + kdf, 1e3 + kdf], dtype=float)
        try:
            ev = evaluate_closure_point(
                root_config,
                model=model,
                theta_B_deg=theta_deg,
                kdf=kdf,
                fan_turn_deg=fan_turn_deg,
            )
            return np.asarray(
                [ev["radius_residual"], ev["length_residual"]],
                dtype=float,
            )
        except Exception:
            return np.array([1e3, 1e3], dtype=float)

    try:
        from scipy.optimize import root
    except Exception as exc:  # pragma: no cover - depends on minimal envs
        ev = evaluate_closure_point(
            root_config,
            model=model,
            theta_B_deg=theta_B_seed_deg,
            kdf=kdf_seed,
            fan_turn_deg=fan_turn_deg,
        )
        return _root_result_from_eval(model, f"scipy unavailable: {exc}", False, ev)

    seed = np.asarray([float(theta_B_seed_deg), float(kdf_seed)], dtype=float)
    sol = root(residual, seed, method="hybr", options={"maxfev": int(maxfev)})
    theta_star, kdf_star = (float(sol.x[0]), float(sol.x[1]))
    ev = evaluate_closure_point(
        root_config,
        model=model,
        theta_B_deg=theta_star,
        kdf=kdf_star,
        fan_turn_deg=fan_turn_deg,
    )
    geom_ok = float(np.linalg.norm(residual(sol.x))) < 1e-6
    return _root_result_from_eval(
        model,
        str(sol.message),
        bool(sol.success and geom_ok),
        ev,
    )


def refine_from_scan_best(
    result: ExistenceScanResult,
    *,
    model: str,
    maxfev: int = 80,
) -> RootRefineResult:
    """Root-refine a model from the best cell in an existing scan."""
    summary = result.closures[model].best_summary()
    return refine_geometry_root(
        result.config,
        model=model,
        theta_B_seed_deg=float(summary["theta_B_deg"]),
        kdf_seed=float(summary["kdf"]),
        fan_turn_deg=float(summary.get("fan_turn_deg", 0.0)),
        maxfev=maxfev,
    )


def resolution_convergence(
    config: ExistenceScanConfig,
    *,
    n_kernel_values: Iterable[int],
    model: str = MODEL_SMOOTH,
    theta_B_seed_deg: float,
    kdf_seed: float,
    fan_turn_deg: float = 0.0,
    maxfev: int = 80,
) -> list[RootRefineResult]:
    """Run root refinement across kernel resolutions, warm-starting each row."""
    rows: list[RootRefineResult] = []
    theta_seed = float(theta_B_seed_deg)
    kdf_seed_val = float(kdf_seed)
    for n_kernel in n_kernel_values:
        cfg = replace(config, n_kernel=int(n_kernel))
        row = refine_geometry_root(
            cfg,
            model=model,
            theta_B_seed_deg=theta_seed,
            kdf_seed=kdf_seed_val,
            fan_turn_deg=fan_turn_deg,
            maxfev=maxfev,
        )
        rows.append(row)
        if row.success and math.isfinite(row.theta_B_deg) and math.isfinite(row.kdf):
            theta_seed = row.theta_B_deg
            kdf_seed_val = row.kdf
    return rows


def write_root_results(
    roots: Mapping[str, RootRefineResult] | Iterable[RootRefineResult],
    output_dir: str | Path,
    *,
    filename: str = "root_refine.json",
) -> None:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    if isinstance(roots, Mapping):
        payload = {name: root.to_dict() for name, root in roots.items()}
    else:
        payload = [root.to_dict() for root in roots]
    with (outdir / filename).open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _empty_closure_arrays(model: str, theta_axis: np.ndarray, kdf_axis: np.ndarray) -> ClosureScanResult:
    shape = (len(theta_axis), len(kdf_axis))
    nan = np.full(shape, np.nan, dtype=float)
    return ClosureScanResult(
        model=model,
        theta_b_deg=np.asarray(theta_axis, dtype=float),
        kdf=np.asarray(kdf_axis, dtype=float),
        residual_norm=nan.copy(),
        radius_residual=nan.copy(),
        length_residual=nan.copy(),
        sigma_E_rad=nan.copy(),
        theta_rao_E_deg=nan.copy(),
        mass_residual=nan.copy(),
        performance_residual=nan.copy(),
        rao_exit_residual=nan.copy(),
        thrust_coefficient=nan.copy(),
        thrust_coefficient_target=nan.copy(),
        d_mach_pre=nan.copy(),
        d_mach_post=nan.copy(),
        d_mach_jump=nan.copy(),
        d_theta_pre_deg=nan.copy(),
        d_theta_post_deg=nan.copy(),
        d_theta_jump_deg=nan.copy(),
        fan_turn_deg=nan.copy(),
        exit_angle_deg=nan.copy(),
        exit_mach=nan.copy(),
        rao_eq5_evaluable=np.zeros(shape, dtype=bool),
        rao_eq5_valid=np.zeros(shape, dtype=bool),
        rao_eq5_min_dr_over_r=nan.copy(),
        success=np.zeros(shape, dtype=bool),
        failure_code=np.full(shape, 6, dtype=int),
    )


def scan_existence(config: ExistenceScanConfig) -> ExistenceScanResult:
    """Run the solver-independent ``(theta_B, kdf)`` existence scan."""
    if config.Rt <= 0.0:
        raise ValueError("Rt must be positive")
    if config.epsilon <= 1.0:
        raise ValueError("epsilon must be > 1")
    if config.stop_at not in STOP_MODES:
        raise ValueError(f"stop_at must be one of {STOP_MODES}")
    if config.scan_mode not in SCAN_MODES:
        raise ValueError(f"scan_mode must be one of {SCAN_MODES}")
    unknown = set(config.models).difference(MODELS)
    if unknown:
        raise ValueError(f"unknown closure model(s): {sorted(unknown)}")

    theta_axis = config.theta_axis
    kdf_axis = config.kdf_axis
    out = ExistenceScanResult(config=config)
    closures = {
        model: _empty_closure_arrays(model, theta_axis, kdf_axis)
        for model in config.models
    }
    cf_target = ideal_exit_cf(config)

    for i, theta_deg in enumerate(theta_axis):
        try:
            kernel = build_kernel(
                config.Rt,
                config.throat_downstream_radius_factor * config.Rt,
                math.radians(float(theta_deg)),
                config.gamma,
                config.n_kernel,
                starting_line_method=config.starting_line_method,
                Ru=config.throat_upstream_radius_factor * config.Rt,
            )
            kernel_bd = tuple(node.to_flow_node() for node in kernel.bd)
        except Exception:
            for closure in closures.values():
                closure.failure_code[i, :] = 1
            continue

        for j, kdf in enumerate(kdf_axis):
            try:
                mass_BD, bd_segment = calc_mdot_bd(kernel_bd, float(kdf), config.gamma)
                if mass_BD <= 0.0 or not bd_segment:
                    raise ValueError("nonpositive BD mass")
                p = bd_segment[-1]
                pre_D = MOCNode(
                    float(p.x), float(p.r), max(float(p.M), 1.000001),
                    float(p.theta), config.gamma,
                )
            except Exception:
                for closure in closures.values():
                    closure.failure_code[i, j] = 2
                continue

            for model, closure in closures.items():
                best: dict | None = None
                best_turn = float("nan")
                for post, turn_deg, code in _post_state_candidates(pre_D, model, config):
                    if code != 0 or post is None:
                        continue
                    cand = _evaluate_candidate(pre_D, post, mass_BD, config, cf_target)
                    if best is None or cand.get("residual_norm", float("inf")) < best.get("residual_norm", float("inf")):
                        best = cand
                        best_turn = float(turn_deg)

                if best is None:
                    closure.failure_code[i, j] = 6
                    continue

                closure.residual_norm[i, j] = best.get("residual_norm", np.nan)
                closure.radius_residual[i, j] = best.get("radius_residual", np.nan)
                closure.length_residual[i, j] = best.get("length_residual", np.nan)
                closure.sigma_E_rad[i, j] = best.get("sigma_E_rad", np.nan)
                closure.theta_rao_E_deg[i, j] = best.get("theta_rao_E_deg", np.nan)
                closure.mass_residual[i, j] = best.get("mass_residual", np.nan)
                closure.performance_residual[i, j] = best.get("performance_residual", np.nan)
                closure.rao_exit_residual[i, j] = best.get("rao_exit_residual", np.nan)
                closure.thrust_coefficient[i, j] = best.get("thrust_coefficient", np.nan)
                closure.thrust_coefficient_target[i, j] = best.get("thrust_coefficient_target", np.nan)
                closure.d_mach_pre[i, j] = best.get("d_mach_pre", np.nan)
                closure.d_mach_post[i, j] = best.get("d_mach_post", np.nan)
                closure.d_mach_jump[i, j] = best.get("d_mach_jump", np.nan)
                closure.d_theta_pre_deg[i, j] = best.get("d_theta_pre_deg", np.nan)
                closure.d_theta_post_deg[i, j] = best.get("d_theta_post_deg", np.nan)
                closure.d_theta_jump_deg[i, j] = best.get("d_theta_jump_deg", np.nan)
                closure.fan_turn_deg[i, j] = best_turn
                closure.exit_angle_deg[i, j] = best.get("exit_angle_deg", np.nan)
                closure.exit_mach[i, j] = best.get("exit_mach", np.nan)
                closure.rao_eq5_evaluable[i, j] = bool(
                    best.get("rao_eq5_evaluable", False)
                )
                closure.rao_eq5_valid[i, j] = bool(
                    best.get("rao_eq5_valid", False)
                )
                closure.rao_eq5_min_dr_over_r[i, j] = best.get(
                    "rao_eq5_min_dr_over_r", np.nan
                )
                closure.success[i, j] = bool(best.get("success", False))
                closure.failure_code[i, j] = int(best.get("failure_code", 0))

    out.closures = closures
    return out


def _scan_source_sha256(
    result: ExistenceScanResult,
    *,
    model: str,
) -> str:
    """Hash the exact fixed-design scan evidence used by one domain sample."""
    closure = result.closures[model]
    config_payload = {
        name: _json_safe(getattr(result.config, name))
        for name in result.config.__dataclass_fields__
    }
    closure_payload = {
        name: _json_safe(getattr(closure, name))
        for name in closure.__dataclass_fields__
    }
    return _payload_sha256(
        {
            "closure": closure_payload,
            "config": config_payload,
            "evidence_model_id": (
                "raosim.rao_existence_scan:fixed-design-scan-evidence:v1"
            ),
            "model": model,
        }
    )


def _failed_scan_attempt_sha256(
    config: ExistenceScanConfig,
    *,
    model: str,
    error: Exception,
) -> str:
    """Hash a failed scan attempt without pretending a scan result exists."""
    config_payload = {
        name: _json_safe(getattr(config, name))
        for name in config.__dataclass_fields__
    }
    return _payload_sha256(
        {
            "attempt_model_id": (
                "raosim.rao_existence_scan:failed-scan-attempt:v1"
            ),
            "config": config_payload,
            "error_message": str(error),
            "error_type": type(error).__name__,
            "model": model,
        }
    )


def _kernel_resolution_evidence(
    rows: Iterable[RootRefineResult],
    *,
    n_kernel_finest: int,
    theta_delta_tolerance_deg: float,
    kdf_delta_tolerance: float,
) -> RaoKernelResolutionEvidence:
    rows = tuple(rows)
    n_kernel = tuple(int(row.n_kernel) for row in rows)
    successes = tuple(bool(row.success) for row in rows)
    theta_values = tuple(
        _optional_finite_float(row.theta_B_deg) for row in rows
    )
    kdf_values = tuple(_optional_finite_float(row.kdf) for row in rows)
    residual_values = tuple(
        _optional_finite_float(row.residual_norm_geometry) for row in rows
    )

    if len(rows) < 2:
        status = RESOLUTION_NOT_EVALUATED
        reason = (
            "at least two increasing n_kernel levels are required to declare "
            "kernel resolution"
        )
    elif n_kernel[-1] != int(n_kernel_finest):
        status = RESOLUTION_FAILED
        reason = "the final refinement row does not match n_kernel_finest"
    elif not all(successes):
        status = RESOLUTION_FAILED
        reason = "one or more kernel-refinement roots did not converge"
    elif any(
        value is None
        for value in (
            theta_values[-2],
            theta_values[-1],
            kdf_values[-2],
            kdf_values[-1],
            residual_values[-1],
        )
    ):
        status = RESOLUTION_FAILED
        reason = "final kernel-refinement evidence contains non-finite values"
    else:
        theta_delta = abs(theta_values[-1] - theta_values[-2])
        kdf_delta = abs(kdf_values[-1] - kdf_values[-2])
        if (
            theta_delta <= theta_delta_tolerance_deg
            and kdf_delta <= kdf_delta_tolerance
        ):
            status = RESOLUTION_PASSED
            reason = (
                "the final two kernel refinements converge within the declared "
                "theta_B and kdf tolerances"
            )
        else:
            status = RESOLUTION_FAILED
            reason = (
                "the final two kernel refinements exceed the declared theta_B "
                "or kdf tolerance"
            )

    return RaoKernelResolutionEvidence(
        n_kernel=n_kernel,
        root_success=successes,
        theta_b_deg=theta_values,
        kdf=kdf_values,
        residual_norm_geometry=residual_values,
        status=status,
        reason=reason,
        theta_delta_tolerance_deg=float(theta_delta_tolerance_deg),
        kdf_delta_tolerance=float(kdf_delta_tolerance),
    )


def _unevaluated_resolution(
    reason: str,
    *,
    theta_delta_tolerance_deg: float,
    kdf_delta_tolerance: float,
) -> RaoKernelResolutionEvidence:
    return RaoKernelResolutionEvidence(
        n_kernel=(),
        root_success=(),
        theta_b_deg=(),
        kdf=(),
        residual_norm_geometry=(),
        status=RESOLUTION_NOT_EVALUATED,
        reason=reason,
        theta_delta_tolerance_deg=float(theta_delta_tolerance_deg),
        kdf_delta_tolerance=float(kdf_delta_tolerance),
    )


def scan_convergence_domain(
    base_config: ExistenceScanConfig,
    *,
    epsilon_values: Sequence[float],
    length_pct_values: Sequence[float],
    model: str = MODEL_SMOOTH,
    n_kernel_values: Sequence[int] = (),
    residual_norm_limit: float = 1.0e-6,
    theta_delta_tolerance_deg: float = 5.0e-2,
    kdf_delta_tolerance: float = 1.0e-3,
    max_epsilon_spacing: float,
    max_length_pct_spacing: float,
    root_maxfev: int = 80,
    theta_b_seed_policy: str = THETA_SEED_FIXED,
    theta_b_chart_span_deg: float = 0.0,
    theta_b_chart_count: int = 0,
) -> RaoConvergenceDomain:
    """Build a declared provider domain over ``epsilon x length_pct``.

    Each external design point first uses :func:`scan_existence` for a root
    seed, then refines the fixed-end geometry across ``n_kernel_values``.  The
    final root is independently evaluated for closure residual and Rao 1999
    Eq. (5) validity.  Fewer than two kernel levels deliberately produce an
    under-resolved artifact rather than implied support.

    The scan is nondimensional in ``Rt``: a uniformly scaled throat changes
    dimensional coordinates and mass flux but not the normalized roots,
    residuals, or Eq. (5) ``dR/R`` verdict.  ``Rt`` is therefore retained as
    reference provenance in :class:`RaoConvergenceContext` but is excluded
    from context matching.
    """
    epsilon_axis = _strict_axis("epsilon_values", epsilon_values)
    length_axis = _strict_axis("length_pct_values", length_pct_values)
    if any(value <= 1.0 for value in epsilon_axis):
        raise ValueError("all epsilon values must be > 1")
    if any(value <= 0.0 for value in length_axis):
        raise ValueError("all length_pct values must be positive")
    if model not in (MODEL_SMOOTH, MODEL_FAN):
        raise ValueError(
            "declared-domain root refinement supports smooth and fan closures"
        )
    residual_limit = _require_finite(
        "residual_norm_limit", residual_norm_limit
    )
    if residual_limit <= 0.0:
        raise ValueError("residual_norm_limit must be positive")
    for name, value in (
        ("theta_delta_tolerance_deg", theta_delta_tolerance_deg),
        ("kdf_delta_tolerance", kdf_delta_tolerance),
        ("max_epsilon_spacing", max_epsilon_spacing),
        ("max_length_pct_spacing", max_length_pct_spacing),
    ):
        checked = _require_finite(name, value)
        if checked <= 0.0:
            raise ValueError(f"{name} must be positive")
    if int(root_maxfev) < 1:
        raise ValueError("root_maxfev must be positive")
    if theta_b_seed_policy not in THETA_SEED_POLICIES:
        raise ValueError(
            f"theta_b_seed_policy must be one of {THETA_SEED_POLICIES}"
        )
    if theta_b_seed_policy == THETA_SEED_RAO_CHART:
        if int(theta_b_chart_count) < 2:
            raise ValueError(
                "chart-centered theta seeding requires theta_b_chart_count >= 2"
            )
        if (
            not math.isfinite(float(theta_b_chart_span_deg))
            or float(theta_b_chart_span_deg) <= 0.0
        ):
            raise ValueError(
                "chart-centered theta seeding requires a positive finite span"
            )
    elif theta_b_chart_count != 0 or theta_b_chart_span_deg != 0.0:
        raise ValueError(
            "chart theta controls require theta_b_seed_policy='rao_chart_centered'"
        )

    kernel_levels = tuple(int(value) for value in n_kernel_values)
    if not kernel_levels:
        kernel_levels = (int(base_config.n_kernel),)
    if tuple(sorted(set(kernel_levels))) != kernel_levels:
        raise ValueError("n_kernel_values must be strictly increasing")
    if any(value < 5 for value in kernel_levels):
        raise ValueError("all n_kernel_values must be >= 5")
    finest = kernel_levels[-1]
    context = RaoConvergenceContext.from_scan_config(
        base_config,
        closure_model=model,
        n_kernel_finest=finest,
        n_kernel_ladder=kernel_levels,
        root_maxfev=int(root_maxfev),
        theta_b_seed_policy=theta_b_seed_policy,
        theta_b_chart_span_deg=float(theta_b_chart_span_deg),
        theta_b_chart_count=int(theta_b_chart_count),
    )

    sample_rows: list[tuple[RaoConvergenceSample, ...]] = []
    for epsilon in epsilon_axis:
        row_samples: list[RaoConvergenceSample] = []
        for length_pct in length_axis:
            theta_seed_axis = base_config.theta_b_values_deg
            if theta_b_seed_policy == THETA_SEED_RAO_CHART:
                from raosim.nozzle_geometry import lookup_angles

                theta_center_deg, _ = lookup_angles(
                    float(epsilon), float(length_pct)
                )
                theta_seed_axis = np.linspace(
                    float(theta_center_deg)
                    - 0.5 * float(theta_b_chart_span_deg),
                    float(theta_center_deg)
                    + 0.5 * float(theta_b_chart_span_deg),
                    int(theta_b_chart_count),
                )
            config = replace(
                base_config,
                epsilon=float(epsilon),
                length_pct=float(length_pct),
                # Seed on the inexpensive first rung; the refinement ladder
                # below owns promotion to the declared finest kernel mesh.
                n_kernel=kernel_levels[0],
                models=(model,),
                theta_b_values_deg=theta_seed_axis,
            )
            resolution = _unevaluated_resolution(
                "root refinement did not start",
                theta_delta_tolerance_deg=theta_delta_tolerance_deg,
                kdf_delta_tolerance=kdf_delta_tolerance,
            )
            theta_star = kdf_star = residual = minimum_dr = None
            eq5_evaluable = False
            eq5_valid = False
            classification = SAMPLE_NUMERICAL_FAILURE
            reason = "existence scan did not produce an evaluable root seed"

            try:
                scan_result = scan_existence(config)
                source_hash = _scan_source_sha256(scan_result, model=model)
            except Exception as exc:
                source_hash = _failed_scan_attempt_sha256(
                    config, model=model, error=exc
                )
                reason = (
                    "existence scan raised "
                    f"{type(exc).__name__}: {exc}; this numerical failure is "
                    "not evidence of physical non-existence"
                )
                row_samples.append(
                    RaoConvergenceSample(
                        epsilon=float(epsilon),
                        length_pct=float(length_pct),
                        classification=classification,
                        reason=reason,
                        theta_b_deg=None,
                        kdf=None,
                        residual_norm_geometry=None,
                        rao_eq5_evaluable=False,
                        rao_eq5_valid=False,
                        rao_eq5_min_dr_over_r=None,
                        resolution=resolution,
                        source_scan_sha256=source_hash,
                    )
                )
                continue

            try:
                seed = refine_from_scan_best(
                    scan_result, model=model, maxfev=int(root_maxfev)
                )
            except Exception as exc:
                reason = (
                    "geometry-root refinement raised "
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                try:
                    resolution_rows = resolution_convergence(
                        config,
                        n_kernel_values=kernel_levels,
                        model=model,
                        theta_B_seed_deg=float(seed.theta_B_deg),
                        kdf_seed=float(seed.kdf),
                        fan_turn_deg=float(seed.fan_turn_deg),
                        maxfev=int(root_maxfev),
                    )
                    resolution = _kernel_resolution_evidence(
                        resolution_rows,
                        n_kernel_finest=finest,
                        theta_delta_tolerance_deg=theta_delta_tolerance_deg,
                        kdf_delta_tolerance=kdf_delta_tolerance,
                    )
                    final_root = resolution_rows[-1] if resolution_rows else seed
                    theta_star = _optional_finite_float(final_root.theta_B_deg)
                    kdf_star = _optional_finite_float(final_root.kdf)
                    residual = _optional_finite_float(
                        final_root.residual_norm_geometry
                    )
                    final_config = replace(config, n_kernel=int(final_root.n_kernel))
                    evaluation = evaluate_closure_point(
                        final_config,
                        model=model,
                        theta_B_deg=float(final_root.theta_B_deg),
                        kdf=float(final_root.kdf),
                        fan_turn_deg=float(final_root.fan_turn_deg),
                    )
                    eq5_evaluable = bool(
                        evaluation.get("rao_eq5_evaluable", False)
                    )
                    eq5_valid = bool(evaluation.get("rao_eq5_valid", False))
                    minimum_dr = _optional_finite_float(
                        evaluation.get("rao_eq5_min_dr_over_r")
                    )
                except Exception as exc:
                    classification = SAMPLE_NUMERICAL_FAILURE
                    reason = (
                        "resolution/final-root evaluation raised "
                        f"{type(exc).__name__}: {exc}"
                    )
                else:
                    if (
                        not final_root.success
                        or residual is None
                        or residual > residual_limit
                    ):
                        classification = SAMPLE_NOT_CONVERGED
                        reason = (
                            "geometry root did not satisfy the declared closure-"
                            f"residual limit {residual_limit:.17g}"
                        )
                    elif not eq5_evaluable:
                        classification = SAMPLE_UNEVALUABLE
                        reason = "Rao Eq. (5) was unevaluable at the converged root"
                    elif resolution.status != RESOLUTION_PASSED:
                        classification = SAMPLE_UNDER_RESOLVED
                        reason = resolution.reason
                    elif eq5_valid:
                        classification = SAMPLE_CONVERGED_VALID
                        reason = (
                            "geometry root, kernel refinement, and Rao Eq. (5) "
                            "all pass"
                        )
                    else:
                        classification = SAMPLE_CONVERGED_INVALID
                        reason = (
                            "geometry root and kernel refinement pass, but Rao "
                            "Eq. (5) is below its valid-region boundary"
                        )

            row_samples.append(
                RaoConvergenceSample(
                    epsilon=float(epsilon),
                    length_pct=float(length_pct),
                    classification=classification,
                    reason=reason,
                    theta_b_deg=theta_star,
                    kdf=kdf_star,
                    residual_norm_geometry=residual,
                    rao_eq5_evaluable=eq5_evaluable,
                    rao_eq5_valid=eq5_valid,
                    rao_eq5_min_dr_over_r=minimum_dr,
                    resolution=resolution,
                    source_scan_sha256=source_hash,
                )
            )
        sample_rows.append(tuple(row_samples))

    return RaoConvergenceDomain(
        context=context,
        epsilon_axis=epsilon_axis,
        length_pct_axis=length_axis,
        samples=tuple(sample_rows),
        residual_norm_limit=residual_limit,
        theta_delta_tolerance_deg=float(theta_delta_tolerance_deg),
        kdf_delta_tolerance=float(kdf_delta_tolerance),
        max_epsilon_spacing=float(max_epsilon_spacing),
        max_length_pct_spacing=float(max_length_pct_spacing),
    )


def write_scan_tables(result: ExistenceScanResult, output_dir: str | Path) -> None:
    """Write one CSV table per closure plus a JSON summary."""
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(result.summary(), f, indent=2, sort_keys=True)

    for model, closure in result.closures.items():
        path = outdir / f"{model}_scan.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "theta_B_deg", "kdf", "residual_norm",
                "radius_residual", "length_residual", "sigma_E_rad",
                "theta_Rao_E_deg", "mass_residual", "performance_residual",
                "rao_exit_residual", "thrust_coefficient",
                "thrust_coefficient_target",
                "D_M_pre", "D_M_post", "D_M_jump",
                "D_theta_pre_deg", "D_theta_post_deg", "D_theta_jump_deg",
                "fan_turn_deg", "exit_angle_deg", "exit_mach",
                "rao_eq5_evaluable", "rao_eq5_valid",
                "rao_eq5_min_dr_over_r",
                "success", "failure_code", "failure_reason",
            ])
            for i, theta_deg in enumerate(closure.theta_b_deg):
                for j, kdf in enumerate(closure.kdf):
                    code = int(closure.failure_code[i, j])
                    writer.writerow([
                        float(theta_deg),
                        float(kdf),
                        float(closure.residual_norm[i, j]),
                        float(closure.radius_residual[i, j]),
                        float(closure.length_residual[i, j]),
                        float(closure.sigma_E_rad[i, j]),
                        float(closure.theta_rao_E_deg[i, j]),
                        float(closure.mass_residual[i, j]),
                        float(closure.performance_residual[i, j]),
                        float(closure.rao_exit_residual[i, j]),
                        float(closure.thrust_coefficient[i, j]),
                        float(closure.thrust_coefficient_target[i, j]),
                        float(closure.d_mach_pre[i, j]),
                        float(closure.d_mach_post[i, j]),
                        float(closure.d_mach_jump[i, j]),
                        float(closure.d_theta_pre_deg[i, j]),
                        float(closure.d_theta_post_deg[i, j]),
                        float(closure.d_theta_jump_deg[i, j]),
                        float(closure.fan_turn_deg[i, j]),
                        float(closure.exit_angle_deg[i, j]),
                        float(closure.exit_mach[i, j]),
                        bool(closure.rao_eq5_evaluable[i, j]),
                        bool(closure.rao_eq5_valid[i, j]),
                        float(closure.rao_eq5_min_dr_over_r[i, j]),
                        bool(closure.success[i, j]),
                        code,
                        FAILURE_CODES.get(code, "unknown"),
                    ])


def _plot_one_heatmap(ax, closure: ClosureScanResult, values: np.ndarray, title: str):
    extent = [
        float(closure.kdf[0]),
        float(closure.kdf[-1]),
        float(closure.theta_b_deg[0]),
        float(closure.theta_b_deg[-1]),
    ]
    data = np.asarray(values, dtype=float)
    finite = np.where(np.isfinite(data), data, np.nan)
    image = ax.imshow(
        finite,
        origin="lower",
        aspect="auto",
        extent=extent,
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel("kdf")
    ax.set_ylabel("theta_B [deg]")
    return image


def plot_scan_heatmaps(result: ExistenceScanResult, output_dir: str | Path) -> None:
    """Create heatmaps for residuals, D Mach behavior, exit angle, failures."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    for model, closure in result.closures.items():
        fields = [
            ("norm", closure.residual_norm),
            ("radius", closure.radius_residual),
            ("length", closure.length_residual),
            ("sigma_E_rad", closure.sigma_E_rad),
            ("mass", closure.mass_residual),
            ("performance_diag", closure.performance_residual),
            ("D_M_jump", closure.d_mach_jump),
            ("exit_angle", closure.exit_angle_deg),
            ("failure_code", closure.failure_code.astype(float)),
        ]
        fig, axes = plt.subplots(3, 3, figsize=(16, 10), constrained_layout=True)
        for ax, (name, values) in zip(axes.ravel(), fields):
            image = _plot_one_heatmap(ax, closure, values, name)
            fig.colorbar(image, ax=ax, shrink=0.86)
        fig.suptitle(f"{model} D-attachment scan")
        fig.savefig(outdir / f"{model}_heatmaps.png", dpi=160)
        plt.close(fig)


__all__ = [
    "CONVERGENCE_DOMAIN_SCHEMA_VERSION",
    "CONVERGENCE_DOMAIN_SOURCE_MODEL_ID",
    "ClosureScanResult",
    "DEIntegrationResult",
    "DOMAIN_CONTEXT_MISMATCH",
    "DOMAIN_LOOKUP_STATUSES",
    "DOMAIN_MODEL_MISMATCH",
    "DOMAIN_REFUSED_INVALID_RAO_REGION",
    "DOMAIN_REFUSED_NOT_CONVERGED",
    "DOMAIN_SUPPORTED",
    "DOMAIN_UNKNOWN_OUTSIDE",
    "DOMAIN_UNKNOWN_SCAN_FAILURE",
    "DOMAIN_UNKNOWN_UNDER_RESOLVED",
    "ExistenceScanConfig",
    "ExistenceScanResult",
    "FAILURE_CODES",
    "MODEL_FAN",
    "MODEL_POSITION",
    "MODEL_SMOOTH",
    "MODELS",
    "RaoConvergenceContext",
    "RaoConvergenceDomain",
    "RaoConvergenceDomainError",
    "RaoConvergenceSample",
    "RaoDomainLookup",
    "RaoKernelResolutionEvidence",
    "RAO_CHART_THETA_SEED_SOURCE_ID",
    "RESOLUTION_FAILED",
    "RESOLUTION_NOT_EVALUATED",
    "RESOLUTION_PASSED",
    "RESOLUTION_STATUSES",
    "RootRefineResult",
    "SCAN_DIAGNOSTIC",
    "SCAN_GEOMETRY",
    "SCAN_MODES",
    "SCAN_STATIONARITY",
    "STOP_LENGTH",
    "STOP_MASS",
    "STOP_RADIUS",
    "THETA_SEED_FIXED",
    "THETA_SEED_POLICIES",
    "THETA_SEED_RAO_CHART",
    "SAMPLE_CLASSIFICATIONS",
    "SAMPLE_CONVERGED_INVALID",
    "SAMPLE_CONVERGED_VALID",
    "SAMPLE_NOT_CONVERGED",
    "SAMPLE_NUMERICAL_FAILURE",
    "SAMPLE_UNDER_RESOLVED",
    "SAMPLE_UNEVALUABLE",
    "evaluate_closure_point",
    "integrate_de",
    "plot_scan_heatmaps",
    "prandtl_meyer_fan_post_state",
    "refine_from_scan_best",
    "refine_geometry_root",
    "resolution_convergence",
    "scan_convergence_domain",
    "scan_existence",
    "sigma_E_rad",
    "stationarity_derivation_summary",
    "target_length",
    "write_root_results",
    "write_scan_tables",
]
