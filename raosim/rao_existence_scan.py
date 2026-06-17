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
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Mapping

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
                closure.success[i, j] = bool(best.get("success", False))
                closure.failure_code[i, j] = int(best.get("failure_code", 0))

    out.closures = closures
    return out


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
    "ClosureScanResult",
    "DEIntegrationResult",
    "ExistenceScanConfig",
    "ExistenceScanResult",
    "FAILURE_CODES",
    "MODEL_FAN",
    "MODEL_POSITION",
    "MODEL_SMOOTH",
    "MODELS",
    "RootRefineResult",
    "SCAN_DIAGNOSTIC",
    "SCAN_GEOMETRY",
    "SCAN_MODES",
    "SCAN_STATIONARITY",
    "STOP_LENGTH",
    "STOP_MASS",
    "STOP_RADIUS",
    "evaluate_closure_point",
    "integrate_de",
    "plot_scan_heatmaps",
    "prandtl_meyer_fan_post_state",
    "refine_from_scan_best",
    "refine_geometry_root",
    "resolution_convergence",
    "scan_existence",
    "sigma_E_rad",
    "stationarity_derivation_summary",
    "target_length",
    "write_root_results",
    "write_scan_tables",
]
