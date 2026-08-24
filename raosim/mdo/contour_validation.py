"""Host-side validation of an optimized TOP contour against Rao's BVP.

This module deliberately does not implement a second inner-loop
``ContourProvider``.  The differentiable optimizer continues to use the fixed
topology TOP/Bezier wall.  After that optimizer has selected a candidate, this
module can solve the independent Rao variational/MOC boundary-value problem at
the *solved* state and compare the two contours under an explicit adjudication
policy.

The contract is fail closed:

* a convergence-domain artifact must cover the exact solve context;
* the variational solve must satisfy its numerical and MOC closure gates;
* Rao 1999 Eq. (5) must be evaluable and valid;
* performance must come from the direct complete C-D-E surface integral; and
* every geometry/performance discrepancy must be finite.

Missing evidence and failed numerical solves are ``unknown``.  A demonstrated
invalid Rao region or an exceeded discrepancy tolerance is ``fail``.  No error
path substitutes the analytic TOP wall for the variational result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np

from raosim.mdo.nozzle import (
    ANALYTIC_TOP_MODEL_CODE,
    ANALYTIC_TOP_PROVIDER_ID,
)
from raosim.rao_variational import (
    RAO_RADIAL_INCREMENT_SOURCE_ID,
    RaoInvalidRegionError,
    RaoRadialIncrementValidity,
    RaoSolution,
    RaoSolverConfig,
    solve_rao_bvp,
)


VARIATIONAL_CONTOUR_MODEL_ID = "rao_variational_moc_bvp"
VARIATIONAL_VALIDATOR_MODEL_ID = "top_vs_rao_variational_postsolve_v1"
VARIATIONAL_SOLVER_SOURCE_ID = "raosim.rao_variational.solve_rao_bvp"
VARIATIONAL_VALIDATOR_SOURCE_ID = (
    "raosim.mdo.contour_validation.validate_optimized_top_contour"
)
VARIATIONAL_FORMULATION = "characteristic"
VARIATIONAL_GAMMA_MODEL_ID = "calorically_perfect"
CONTOUR_COMPARISON_POLICY_ID = "top-vs-variational-screening-tolerances-v1"
DIRECT_FULL_CDE_CF_BASIS = "direct_full_cde_surface_integral"
ANALYTIC_TOP_CF_BASIS = "mdo_one_dimensional_ideal_exit_momentum_pressure"


class VariationalValidationStatus(str, Enum):
    """Three-valued adjudication result for the post-solve check."""

    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


class VariationalValidationReason(str, Enum):
    """Stable reason identities for validation outcomes."""

    WITHIN_TOLERANCES = "within_tolerances"
    DISCREPANCY_EXCEEDED = "discrepancy_exceeded"
    INVALID_RAO_REGION = "invalid_rao_region"
    RAO_VALIDITY_UNAVAILABLE = "rao_eq5_validity_unavailable"
    CONVERGENCE_DOMAIN_NOT_SUPPLIED = "convergence_domain_not_supplied"
    CONVERGENCE_DOMAIN_LOAD_FAILED = "convergence_domain_load_failed"
    CONVERGENCE_DOMAIN_CONTEXT_MISMATCH = "convergence_domain_context_mismatch"
    CONVERGENCE_DOMAIN_UNSUPPORTED = "convergence_domain_unsupported"
    VARIATIONAL_SOLVE_EXCEPTION = "variational_solve_exception"
    VARIATIONAL_NOT_CONVERGED = "variational_not_converged"
    MOC_OR_PHYSICAL_CLOSURE_FAILED = "moc_or_physical_closure_failed"
    FULL_CDE_THRUST_UNAVAILABLE = "full_cde_thrust_unavailable"
    CONTOUR_EXPORT_REFUSED = "contour_export_refused"
    NONFINITE_OR_MALFORMED_GEOMETRY = "nonfinite_or_malformed_geometry"
    NONFINITE_DISCREPANCY = "nonfinite_discrepancy"
    INVALID_SOLVED_STATE = "invalid_solved_state"


@dataclass(frozen=True)
class ContourDiscrepancyTolerances:
    """Explicit post-solve adjudication thresholds.

    These values are project screening policy, not literature qualification
    limits.  They intentionally have no implicit defaults: a caller must state
    the policy used to flag an optimized candidate.
    """

    max_radius_error_over_rt: float
    rms_radius_error_over_rt: float
    relative_divergent_length_error: float
    relative_thrust_coefficient_error: float

    def __post_init__(self) -> None:
        values = {
            "max_radius_error_over_rt": self.max_radius_error_over_rt,
            "rms_radius_error_over_rt": self.rms_radius_error_over_rt,
            "relative_divergent_length_error": (
                self.relative_divergent_length_error
            ),
            "relative_thrust_coefficient_error": (
                self.relative_thrust_coefficient_error
            ),
        }
        for name, value in values.items():
            if isinstance(value, bool):
                raise ValueError(f"{name} must be a numeric tolerance, not bool")
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            # Canonicalize accepted float-like inputs.  Otherwise a string
            # such as ``"0.01"`` validates here but fails later when the
            # measured discrepancy is compared with the stored threshold.
            object.__setattr__(self, name, numeric)

    def to_dict(self) -> dict[str, float]:
        return {
            "max_radius_error_over_rt": float(self.max_radius_error_over_rt),
            "rms_radius_error_over_rt": float(self.rms_radius_error_over_rt),
            "relative_divergent_length_error": float(
                self.relative_divergent_length_error
            ),
            "relative_thrust_coefficient_error": float(
                self.relative_thrust_coefficient_error
            ),
        }


@runtime_checkable
class RaoConvergenceDomainLike(Protocol):
    """Structural seam implemented by the B5 convergence-domain artifact."""

    def lookup(self, *args: Any, **kwargs: Any) -> Any:
        ...


RaoConvergenceDomainSource = RaoConvergenceDomainLike | str | Path | None


@dataclass(frozen=True)
class VariationalPostSolveConfig:
    """Host solve settings plus the explicit comparison policy.

    ``evaluate_moc`` is intentionally not a switch here: B3 requires the
    complete variational/MOC solve, so the validator always enables it.  Wall
    construction is explicit because the BDE and coupled paths have different
    convergence behavior.  A domain artifact is optional at construction time
    to make missing evidence reportable, but no outcome can pass without one.
    """

    tolerances: ContourDiscrepancyTolerances
    convergence_domain: RaoConvergenceDomainSource = None
    comparison_points: int = 257
    n_control: int = 12
    n_kernel: int = 12
    max_nfev: int = 200
    residual_tol: float = 2.0e-3
    starting_line_method: str = "kliegel_levine"
    solver_backend: str = "jax"
    wall_method: str = "coupled"
    formulation: str = VARIATIONAL_FORMULATION
    physics_weight: float = 1.0
    jax_constraint_weight_ladder: tuple[float, ...] | None = (
        1.0,
        10.0,
        30.0,
        100.0,
    )
    kernel_d_fraction_min: float = 0.0
    kernel_d_fraction_max: float | None = None
    solve_theta_b: bool = False
    rao_radial_increment_tol: float = 1.0e-9

    def comparison_policy_dict(self) -> dict[str, Any]:
        """Return the canonical identity of the four-threshold B3 policy."""

        return {
            "policy_id": CONTOUR_COMPARISON_POLICY_ID,
            **self.tolerances.to_dict(),
        }

    def __post_init__(self) -> None:
        if int(self.comparison_points) < 3:
            raise ValueError("comparison_points must be at least 3")
        if int(self.n_control) < 8:
            raise ValueError("n_control must be at least 8")
        if int(self.n_kernel) < 2:
            raise ValueError("n_kernel must be at least 2")
        if int(self.max_nfev) < 1:
            raise ValueError("max_nfev must be positive")
        if not math.isfinite(float(self.residual_tol)) or self.residual_tol <= 0:
            raise ValueError("residual_tol must be finite and positive")
        if self.solver_backend not in {"jax", "numpy"}:
            raise ValueError("solver_backend must be 'jax' or 'numpy'")
        if self.wall_method not in {"coupled", "legacy", "bde"}:
            raise ValueError("wall_method must be 'coupled', 'legacy', or 'bde'")
        if self.formulation != VARIATIONAL_FORMULATION:
            raise ValueError(
                "B3 variational validation requires the 'characteristic' "
                "formulation"
            )
        if (
            not math.isfinite(float(self.physics_weight))
            or self.physics_weight <= 0.0
        ):
            raise ValueError("physics_weight must be finite and positive")
        ladder = self.jax_constraint_weight_ladder
        if ladder is not None:
            if not ladder:
                raise ValueError(
                    "jax_constraint_weight_ladder must be non-empty or None"
                )
            if any(
                not math.isfinite(float(value)) or float(value) <= 0.0
                for value in ladder
            ):
                raise ValueError(
                    "jax_constraint_weight_ladder values must be finite and positive"
                )
        if not str(self.starting_line_method).strip():
            raise ValueError("starting_line_method must be non-empty")
        lo = float(self.kernel_d_fraction_min)
        hi = self.kernel_d_fraction_max
        if not math.isfinite(lo) or not 0.0 <= lo < 1.0:
            raise ValueError("kernel_d_fraction_min must lie in [0, 1)")
        if hi is not None:
            hi = float(hi)
            if not math.isfinite(hi) or not lo < hi <= 1.0:
                raise ValueError(
                    "kernel_d_fraction_max must exceed the minimum and be <= 1"
                )
        if (
            not math.isfinite(float(self.rao_radial_increment_tol))
            or self.rao_radial_increment_tol < 0.0
        ):
            raise ValueError(
                "rao_radial_increment_tol must be finite and non-negative"
            )


@dataclass(frozen=True)
class ContourDiscrepancy:
    """Measured TOP-versus-variational discrepancies and their thresholds."""

    max_radius_error_over_rt: float
    max_radius_error_over_rt_limit: float
    max_radius_error_m: float
    rms_radius_error_over_rt: float
    rms_radius_error_over_rt_limit: float
    rms_radius_error_m: float
    relative_divergent_length_error: float
    relative_divergent_length_error_limit: float
    relative_thrust_coefficient_error: float
    relative_thrust_coefficient_error_limit: float
    analytic_divergent_length_m: float
    variational_divergent_length_m: float
    analytic_thrust_coefficient: float
    variational_thrust_coefficient: float
    exit_radius_error_m: float
    exit_radius_error_over_rt: float
    exit_wall_angle_delta_deg: float
    adjudication_eligible: bool = True
    adjudication_ineligibility_reason: str | None = None
    analytic_performance_basis: str = ANALYTIC_TOP_CF_BASIS
    variational_performance_basis: str = DIRECT_FULL_CDE_CF_BASIS

    @property
    def geometry_passes(self) -> bool:
        return bool(
            self.max_radius_error_over_rt
            <= self.max_radius_error_over_rt_limit
            and self.rms_radius_error_over_rt
            <= self.rms_radius_error_over_rt_limit
            and self.relative_divergent_length_error
            <= self.relative_divergent_length_error_limit
        )

    @property
    def performance_passes(self) -> bool:
        return bool(
            self.relative_thrust_coefficient_error
            <= self.relative_thrust_coefficient_error_limit
        )

    @property
    def passes(self) -> bool:
        return (
            self.adjudication_eligible
            and self.geometry_passes
            and self.performance_passes
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_radius_error_over_rt": self.max_radius_error_over_rt,
            "max_radius_error_over_rt_limit": self.max_radius_error_over_rt_limit,
            "max_radius_error_m": self.max_radius_error_m,
            "rms_radius_error_over_rt": self.rms_radius_error_over_rt,
            "rms_radius_error_over_rt_limit": self.rms_radius_error_over_rt_limit,
            "rms_radius_error_m": self.rms_radius_error_m,
            "relative_divergent_length_error": (
                self.relative_divergent_length_error
            ),
            "relative_divergent_length_error_limit": (
                self.relative_divergent_length_error_limit
            ),
            "relative_thrust_coefficient_error": (
                self.relative_thrust_coefficient_error
            ),
            "relative_thrust_coefficient_error_limit": (
                self.relative_thrust_coefficient_error_limit
            ),
            "analytic_divergent_length_m": self.analytic_divergent_length_m,
            "variational_divergent_length_m": self.variational_divergent_length_m,
            "analytic_thrust_coefficient": self.analytic_thrust_coefficient,
            "variational_thrust_coefficient": self.variational_thrust_coefficient,
            "exit_radius_error_m": self.exit_radius_error_m,
            "exit_radius_error_over_rt": self.exit_radius_error_over_rt,
            "exit_wall_angle_delta_deg": self.exit_wall_angle_delta_deg,
            "adjudication_eligible": self.adjudication_eligible,
            "adjudication_ineligibility_reason": (
                self.adjudication_ineligibility_reason
            ),
            "analytic_performance_basis": self.analytic_performance_basis,
            "variational_performance_basis": self.variational_performance_basis,
            "geometry_passes": self.geometry_passes,
            "performance_passes": self.performance_passes,
            "passes": self.passes,
        }


@dataclass(frozen=True)
class RaoDomainEvidence:
    """Exact B5 domain identity and lookup result retained by B3."""

    artifact_sha256: str | None
    schema_version: int | None
    source_model_id: str | None
    artifact_context_sha256: str | None
    expected_context_sha256: str | None
    lookup_status: str
    supported: bool | None
    reason: str | None
    context: Mapping[str, Any]
    lookup: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "schema_version": self.schema_version,
            "source_model_id": self.source_model_id,
            "artifact_context_sha256": self.artifact_context_sha256,
            "expected_context_sha256": self.expected_context_sha256,
            "lookup_status": self.lookup_status,
            "supported": self.supported,
            "reason": self.reason,
            "context": dict(self.context),
            "lookup": dict(self.lookup),
        }


@dataclass(frozen=True)
class VariationalContourValidationOutcome:
    """Typed B3 outcome; absence is never represented by a fabricated wall."""

    available: bool
    status: VariationalValidationStatus
    reason: VariationalValidationReason
    detail: str
    analytic_model_id: str = ANALYTIC_TOP_PROVIDER_ID
    variational_model_id: str = VARIATIONAL_CONTOUR_MODEL_ID
    validator_model_id: str = VARIATIONAL_VALIDATOR_MODEL_ID
    solved_state: Mapping[str, Any] | None = None
    domain: RaoDomainEvidence | None = None
    rao_validity: RaoRadialIncrementValidity | None = None
    discrepancy: ContourDiscrepancy | None = None
    numerical_converged: bool | None = None
    physical_moc_closed: bool | None = None
    residual_max_scaled: float | None = None
    reliability: str | None = None
    warnings: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.available and self.status is VariationalValidationStatus.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "status": self.status.value,
            "reason": self.reason.value,
            "detail": self.detail,
            "passed": self.passed,
            "analytic_model_id": self.analytic_model_id,
            "variational_model_id": self.variational_model_id,
            "validator_model_id": self.validator_model_id,
            "solved_state": (
                None if self.solved_state is None else dict(self.solved_state)
            ),
            "domain": None if self.domain is None else self.domain.to_dict(),
            "rao_eq5_source_id": RAO_RADIAL_INCREMENT_SOURCE_ID,
            "rao_validity": (
                None if self.rao_validity is None else self.rao_validity.to_dict()
            ),
            "discrepancy": (
                None if self.discrepancy is None else self.discrepancy.to_dict()
            ),
            "numerical_converged": self.numerical_converged,
            "physical_moc_closed": self.physical_moc_closed,
            "residual_max_scaled": self.residual_max_scaled,
            "reliability": self.reliability,
            "warnings": list(self.warnings),
        }


def _host_float(value: Any, name: str, *, positive: bool = False) -> float:
    try:
        out = float(np.asarray(value))
    except Exception as exc:
        raise ValueError(f"{name} is not a scalar") from exc
    if not math.isfinite(out) or (positive and out <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return out


def _callable_source_id(value: Any) -> str:
    """Return a replay-visible identity for the injected solver callable."""

    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _divergent_polyline(
    x: Any,
    r: Any,
    *,
    throat_index: int | None,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a finite, strictly increasing throat-to-exit polyline."""

    xx = np.asarray(x, dtype=float).reshape(-1)
    rr = np.asarray(r, dtype=float).reshape(-1)
    if xx.size != rr.size or xx.size < 2:
        raise ValueError(f"{name} x/r arrays must have equal length >= 2")
    if throat_index is not None:
        index = int(throat_index)
        if not 0 <= index < xx.size - 1:
            raise ValueError(f"{name} throat index is outside the polyline")
        xx, rr = xx[index:], rr[index:]
    else:
        scale = max(float(np.max(np.abs(xx))), 1.0)
        mask = xx >= -1.0e-12 * scale
        xx, rr = xx[mask], rr[mask]
    if xx.size < 2 or not np.all(np.isfinite(xx)) or not np.all(np.isfinite(rr)):
        raise ValueError(f"{name} divergent polyline is nonfinite or too short")
    tolerance = 1.0e-12 * max(float(np.max(np.abs(xx))), 1.0)
    if np.any(np.diff(xx) < -tolerance):
        raise ValueError(f"{name} divergent x coordinate is non-monotonic")
    # Remove repeated axial stations deterministically.  The last radius is
    # kept because it is the downstream side of a junction.
    keep = np.concatenate((np.diff(xx) > tolerance, [True]))
    xx, rr = xx[keep], rr[keep]
    if xx.size < 2 or xx[-1] - xx[0] <= tolerance:
        raise ValueError(f"{name} divergent length is not positive")
    return xx, rr


def compare_top_and_variational_contours(
    *,
    analytic_x: Any,
    analytic_r: Any,
    analytic_throat_index: int,
    variational_x: Any,
    variational_r: Any,
    Rt: float,
    analytic_thrust_coefficient: float,
    variational_thrust_coefficient: float,
    tolerances: ContourDiscrepancyTolerances,
    comparison_points: int = 257,
    adjudication_eligible: bool = True,
    adjudication_ineligibility_reason: str | None = None,
) -> ContourDiscrepancy:
    """Compare two divergent walls at common normalized axial stations."""

    throat_radius = _host_float(Rt, "Rt", positive=True)
    cf_top = _host_float(
        analytic_thrust_coefficient, "analytic_thrust_coefficient", positive=True
    )
    cf_var = _host_float(
        variational_thrust_coefficient,
        "variational_thrust_coefficient",
        positive=True,
    )
    if int(comparison_points) < 3:
        raise ValueError("comparison_points must be at least 3")
    if not adjudication_eligible and not adjudication_ineligibility_reason:
        raise ValueError(
            "an ineligible discrepancy report requires an ineligibility reason"
        )
    top_x, top_r = _divergent_polyline(
        analytic_x,
        analytic_r,
        throat_index=analytic_throat_index,
        name="analytic TOP",
    )
    var_x, var_r = _divergent_polyline(
        variational_x,
        variational_r,
        throat_index=None,
        name="variational",
    )
    top_length = float(top_x[-1] - top_x[0])
    var_length = float(var_x[-1] - var_x[0])
    u = np.linspace(0.0, 1.0, int(comparison_points))
    top_u = (top_x - top_x[0]) / top_length
    var_u = (var_x - var_x[0]) / var_length
    radial_error = (
        np.interp(u, var_u, var_r) - np.interp(u, top_u, top_r)
    ) / throat_radius
    max_radius = float(np.max(np.abs(radial_error)))
    rms_radius = float(np.sqrt(np.mean(radial_error * radial_error)))
    length_error = abs(var_length - top_length) / top_length
    cf_error = abs(cf_var - cf_top) / abs(cf_top)
    exit_radius_error = float(var_r[-1] - top_r[-1])
    top_exit_angle = math.atan2(
        float(top_r[-1] - top_r[-2]), float(top_x[-1] - top_x[-2])
    )
    var_exit_angle = math.atan2(
        float(var_r[-1] - var_r[-2]), float(var_x[-1] - var_x[-2])
    )
    exit_angle_delta_deg = math.degrees(var_exit_angle - top_exit_angle)
    values = (
        max_radius,
        rms_radius,
        length_error,
        cf_error,
        exit_radius_error,
        exit_angle_delta_deg,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("one or more contour discrepancies are nonfinite")
    return ContourDiscrepancy(
        max_radius_error_over_rt=max_radius,
        max_radius_error_over_rt_limit=tolerances.max_radius_error_over_rt,
        max_radius_error_m=max_radius * throat_radius,
        rms_radius_error_over_rt=rms_radius,
        rms_radius_error_over_rt_limit=tolerances.rms_radius_error_over_rt,
        rms_radius_error_m=rms_radius * throat_radius,
        relative_divergent_length_error=length_error,
        relative_divergent_length_error_limit=(
            tolerances.relative_divergent_length_error
        ),
        relative_thrust_coefficient_error=cf_error,
        relative_thrust_coefficient_error_limit=(
            tolerances.relative_thrust_coefficient_error
        ),
        analytic_divergent_length_m=top_length,
        variational_divergent_length_m=var_length,
        analytic_thrust_coefficient=cf_top,
        variational_thrust_coefficient=cf_var,
        exit_radius_error_m=exit_radius_error,
        exit_radius_error_over_rt=exit_radius_error / throat_radius,
        exit_wall_angle_delta_deg=exit_angle_delta_deg,
        adjudication_eligible=bool(adjudication_eligible),
        adjudication_ineligibility_reason=(
            None
            if adjudication_eligible
            else str(adjudication_ineligibility_reason)
        ),
    )


def _base_outcome(
    *,
    status: VariationalValidationStatus,
    reason: VariationalValidationReason,
    detail: str,
    solved_state: Mapping[str, Any] | None,
    available: bool = False,
    domain: RaoDomainEvidence | None = None,
    solution: RaoSolution | None = None,
    discrepancy: ContourDiscrepancy | None = None,
) -> VariationalContourValidationOutcome:
    residual = None
    if solution is not None:
        try:
            residual_value = float(solution.residuals.max_scaled)
        except (TypeError, ValueError):
            pass
        else:
            residual = residual_value if math.isfinite(residual_value) else None
    reliability = (
        None if solution is None else str(solution.reliability.value)
    )
    # This field reports forward-MOC/physical closure only.  Full-CDE thrust
    # availability is an independent gate and must not rewrite an otherwise
    # closed MOC result as false.
    physical_closed = (
        None if solution is None else bool(solution.converged)
    )
    return VariationalContourValidationOutcome(
        available=available,
        status=status,
        reason=reason,
        detail=str(detail),
        solved_state=solved_state,
        domain=domain,
        rao_validity=None if solution is None else solution.rao_validity,
        discrepancy=discrepancy,
        numerical_converged=(
            None
            if solution is None
            else bool(solution.control_surface.converged)
        ),
        physical_moc_closed=physical_closed,
        residual_max_scaled=residual,
        reliability=reliability,
        warnings=() if solution is None else tuple(str(w) for w in solution.warnings),
    )


def unavailable_variational_validation(
    detail: str,
    *,
    reason: VariationalValidationReason = (
        VariationalValidationReason.INVALID_SOLVED_STATE
    ),
) -> VariationalContourValidationOutcome:
    """Construct an explicit unknown outcome when a requested check cannot run."""

    return _base_outcome(
        status=VariationalValidationStatus.UNKNOWN,
        reason=reason,
        detail=detail,
        solved_state=None,
    )


# Domain loading/lookup is finalized immediately below once the B5 public type
# is imported.  Keeping these two helpers isolated makes the validator's
# physics and discrepancy checks independent of the artifact serialization.
def _load_domain(source: RaoConvergenceDomainSource) -> Any:
    if source is None:
        return None
    if isinstance(source, (str, Path)):
        from raosim.rao_existence_scan import RaoConvergenceDomain

        return RaoConvergenceDomain.read_json(Path(source))
    if not isinstance(source, RaoConvergenceDomainLike):
        raise TypeError("convergence_domain must implement lookup(...) or be a path")
    return source


def _lookup_domain(
    domain: Any,
    *,
    epsilon: float,
    length_pct: float,
    solver_config: RaoSolverConfig,
) -> RaoDomainEvidence:
    """Preflight the B5 epsilon/length domain at the exact fixed context."""

    from raosim.rao_existence_scan import MODEL_SMOOTH, RaoConvergenceDomain

    if not isinstance(domain, RaoConvergenceDomain):
        raise TypeError(
            "convergence_domain must be a RaoConvergenceDomain artifact"
        )
    # The selected, hash-verified artifact owns its scan/mesh policy.  Replace
    # only quantities shared with this solved request.  Rt_reference records
    # the actual candidate scale but is excluded from B5's identity after its
    # explicit two-scale invariance gate.
    expected_context = replace(
        domain.context,
        Rt_reference=float(solver_config.Rt),
        gamma=float(solver_config.gamma),
        pa_over_p0=float(solver_config.pa_over_p0),
        throat_downstream_radius_factor=float(
            solver_config.throat_downstream_radius_factor
        ),
        throat_upstream_radius_factor=float(
            solver_config.throat_upstream_radius_factor
        ),
        starting_line_method=str(solver_config.starting_line_method),
        closure_model=MODEL_SMOOTH,
        rt_scale_invariant=True,
    )
    lookup = domain.lookup(
        float(epsilon),
        float(length_pct),
        expected_context=expected_context,
    )
    return RaoDomainEvidence(
        artifact_sha256=str(domain.artifact_sha256),
        schema_version=int(domain.schema_version),
        source_model_id=str(domain.source_model_id),
        artifact_context_sha256=str(domain.context.identity_sha256),
        expected_context_sha256=str(expected_context.identity_sha256),
        lookup_status=str(lookup.status),
        supported=bool(lookup.supported),
        reason=str(lookup.reason),
        context={
            "artifact": domain.context.to_dict(),
            "expected": expected_context.to_dict(),
        },
        lookup=lookup.to_dict(),
    )


def _moc_ineligibility_reason(solution: RaoSolution) -> str:
    """Name the failed post-BVP MOC checks without changing their tolerances."""

    diagnostics = solution.construction_diagnostics
    report = diagnostics.get("net_report", {})
    failed: list[str] = []
    if isinstance(report, Mapping):
        failed.extend(
            str(name)
            for name, value in report.items()
            if (
                str(name).endswith("_passes")
                and isinstance(value, (bool, np.bool_))
                and not bool(value)
            )
        )
    if diagnostics.get("moc_compatibility_preserved") is False:
        failed.append("moc_compatibility_preserved")
    failed = list(dict.fromkeys(failed))
    if failed:
        return "MOC closure gate(s) failed: " + ", ".join(failed)
    return (
        "the BVP residual closed, but the forward MOC "
        "closure/tangency/crossing/smoothness gate did not"
    )


def _moc_smoothness_only_failure(solution: RaoSolution) -> bool:
    """Whether a closed BVP missed only the declared MOC smoothness gate."""

    report = solution.construction_diagnostics.get("net_report", {})
    if not isinstance(report, Mapping):
        return False
    failed = {
        str(name)
        for name, value in report.items()
        if (
            str(name).endswith("_passes")
            and isinstance(value, (bool, np.bool_))
            and not bool(value)
        )
    }
    return failed == {"measured_neighbor_smoothness_passes"}


def validate_optimized_top_contour(
    state: Any,
    mission: Any,
    config: VariationalPostSolveConfig,
    *,
    solver: Callable[[RaoSolverConfig], RaoSolution] = solve_rao_bvp,
) -> VariationalContourValidationOutcome:
    """Run the B3 validator at the exact solved MDO candidate.

    ``state`` is expected to be the versioned ``EngineState``.  All live
    quantities are read from it: notably ``Rt``, ``epsilon``, ``Pc``, effective
    ``gamma``, effective O/F, analytic ``Cf_ideal``, and the exact TOP station
    grid.  Only fixed geometry/ambient conventions come from ``mission``.
    """

    try:
        performance = state.performance
        geometry = state.geometry
        Rt = _host_float(performance.Rt, "state.performance.Rt", positive=True)
        epsilon = _host_float(
            performance.epsilon, "state.performance.epsilon", positive=True
        )
        Pc = _host_float(performance.Pc, "state.performance.Pc", positive=True)
        gamma = _host_float(
            performance.gamma, "state.performance.gamma", positive=True
        )
        mixture_ratio = _host_float(performance.OF, "state.performance.OF", positive=True)
        cf_top = _host_float(
            performance.Cf_ideal,
            "state.performance.Cf_ideal",
            positive=True,
        )
        ambient = _host_float(performance.Pa, "state.performance.Pa")
        if ambient < 0.0:
            raise ValueError("state.performance.Pa must be non-negative")
        throat_index = int(np.asarray(geometry.throat_index))
        divergent_model = int(np.asarray(geometry.divergent_model))
        if divergent_model != ANALYTIC_TOP_MODEL_CODE:
            raise ValueError(
                "B3 compares the analytic TOP optimum only; solved geometry "
                f"model code is {divergent_model}, expected "
                f"{ANALYTIC_TOP_MODEL_CODE}"
            )
        station_valid = np.asarray(geometry.station_valid, dtype=bool)
        if station_valid.size == 0 or not np.all(station_valid):
            raise ValueError("analytic TOP provider outcome is not available")
        analytic_x = np.asarray(geometry.x, dtype=float)
        analytic_r = np.asarray(geometry.r, dtype=float)
        length_pct = _host_float(mission.length_pct, "mission.length_pct", positive=True)
        ru_factor = _host_float(
            mission.throat_ru_factor,
            "mission.throat_ru_factor",
            positive=True,
        )
        rd_factor = _host_float(
            mission.throat_rd_factor,
            "mission.throat_rd_factor",
            positive=True,
        )
        convergent_angle = _host_float(
            mission.converging_half_angle_deg,
            "mission.converging_half_angle_deg",
            positive=True,
        )
    except Exception as exc:
        return unavailable_variational_validation(
            f"solved state could not be bound to B3: {type(exc).__name__}: {exc}"
        )

    solved_state = {
        "Rt_m": Rt,
        "epsilon": epsilon,
        "Pc_pa": Pc,
        "Pa_pa": ambient,
        "pa_over_p0": ambient / Pc,
        "gamma": gamma,
        "gamma_model": VARIATIONAL_GAMMA_MODEL_ID,
        "mixture_ratio": mixture_ratio,
        "length_pct": length_pct,
        "throat_upstream_radius_factor": ru_factor,
        "throat_downstream_radius_factor": rd_factor,
        "converging_half_angle_deg": convergent_angle,
        "analytic_thrust_coefficient": cf_top,
        "analytic_contour_model_code": divergent_model,
        "variational_physical_model_id": VARIATIONAL_CONTOUR_MODEL_ID,
        "postsolve_validator_selection_id": VARIATIONAL_VALIDATOR_MODEL_ID,
        "variational_solver_source_id": _callable_source_id(solver),
        "canonical_variational_solver_source_id": (
            VARIATIONAL_SOLVER_SOURCE_ID
        ),
        "postsolve_validator_source_id": VARIATIONAL_VALIDATOR_SOURCE_ID,
        "variational_solver_backend": config.solver_backend,
        "variational_wall_method": config.wall_method,
        "variational_formulation": config.formulation,
        "variational_physics_weight": float(config.physics_weight),
        "variational_constraint_weight_ladder": (
            None
            if config.jax_constraint_weight_ladder is None
            else [
                float(value)
                for value in config.jax_constraint_weight_ladder
            ]
        ),
        "variational_n_control": int(config.n_control),
        "variational_n_kernel": int(config.n_kernel),
        "variational_max_nfev": int(config.max_nfev),
        "variational_residual_tol": float(config.residual_tol),
        "variational_starting_line_method": config.starting_line_method,
        "variational_kernel_d_fraction_min": float(
            config.kernel_d_fraction_min
        ),
        "variational_kernel_d_fraction_max": (
            None
            if config.kernel_d_fraction_max is None
            else float(config.kernel_d_fraction_max)
        ),
        "variational_solve_theta_b": bool(config.solve_theta_b),
        "rao_radial_increment_tol": float(config.rao_radial_increment_tol),
        "comparison_points": int(config.comparison_points),
        "discrepancy_tolerances": config.tolerances.to_dict(),
    }

    if config.convergence_domain is None:
        return _base_outcome(
            status=VariationalValidationStatus.UNKNOWN,
            reason=VariationalValidationReason.CONVERGENCE_DOMAIN_NOT_SUPPLIED,
            detail=(
                "post-solve variational validation requires a declared B5 "
                "convergence-domain artifact; no solve was attempted"
            ),
            solved_state=solved_state,
        )
    try:
        domain = _load_domain(config.convergence_domain)
    except Exception as exc:
        return _base_outcome(
            status=VariationalValidationStatus.UNKNOWN,
            reason=VariationalValidationReason.CONVERGENCE_DOMAIN_LOAD_FAILED,
            detail=f"could not load convergence domain: {type(exc).__name__}: {exc}",
            solved_state=solved_state,
        )
    solver_config = RaoSolverConfig(
        Rt=Rt,
        epsilon=epsilon,
        gamma=gamma,
        pa_over_p0=ambient / Pc,
        length_pct=length_pct,
        throat_upstream_radius_factor=ru_factor,
        throat_downstream_radius_factor=rd_factor,
        n_control=int(config.n_control),
        n_kernel=int(config.n_kernel),
        max_nfev=int(config.max_nfev),
        residual_tol=float(config.residual_tol),
        starting_line_method=str(config.starting_line_method),
        evaluate_moc=True,
        wall_method=str(config.wall_method),
        formulation=str(config.formulation),
        solver_backend=str(config.solver_backend),
        physics_weight=float(config.physics_weight),
        jax_constraint_weight_ladder=config.jax_constraint_weight_ladder,
        kernel_d_fraction_min=float(config.kernel_d_fraction_min),
        kernel_d_fraction_max=config.kernel_d_fraction_max,
        solve_theta_b=bool(config.solve_theta_b),
        rao_radial_increment_tol=float(config.rao_radial_increment_tol),
    )

    # B5 is a preflight declaration over the external design coordinates.  A
    # candidate outside it is never used as an excuse to run the expensive BVP
    # and then reinterpret a failed attempt as physical evidence.
    try:
        domain_evidence = _lookup_domain(
            domain,
            epsilon=epsilon,
            length_pct=length_pct,
            solver_config=solver_config,
        )
    except Exception as exc:
        return _base_outcome(
            status=VariationalValidationStatus.UNKNOWN,
            reason=VariationalValidationReason.CONVERGENCE_DOMAIN_CONTEXT_MISMATCH,
            detail=f"convergence-domain lookup failed: {type(exc).__name__}: {exc}",
            solved_state=solved_state,
        )
    from raosim.rao_existence_scan import (
        DOMAIN_CONTEXT_MISMATCH,
        DOMAIN_MODEL_MISMATCH,
        DOMAIN_REFUSED_INVALID_RAO_REGION,
    )

    if domain_evidence.lookup_status == DOMAIN_REFUSED_INVALID_RAO_REGION:
        return _base_outcome(
            status=VariationalValidationStatus.FAIL,
            reason=VariationalValidationReason.INVALID_RAO_REGION,
            detail=(
                "the residual-converged, mesh-resolved B5 domain classifies "
                "this candidate below the Rao 1999 Eq. (5) boundary; no "
                "variational solve was attempted"
            ),
            solved_state=solved_state,
            domain=domain_evidence,
            available=True,
        )
    if domain_evidence.supported is not True:
        context_mismatch = domain_evidence.lookup_status in {
            DOMAIN_CONTEXT_MISMATCH,
            DOMAIN_MODEL_MISMATCH,
        }
        return _base_outcome(
            status=VariationalValidationStatus.UNKNOWN,
            reason=(
                VariationalValidationReason.CONVERGENCE_DOMAIN_CONTEXT_MISMATCH
                if context_mismatch
                else VariationalValidationReason.CONVERGENCE_DOMAIN_UNSUPPORTED
            ),
            detail=(
                "the declared B5 domain does not support this exact solved "
                f"state ({domain_evidence.lookup_status}); no variational "
                "solve was attempted"
            ),
            solved_state=solved_state,
            domain=domain_evidence,
        )

    try:
        solution = solver(solver_config)
    except Exception as exc:
        return _base_outcome(
            status=VariationalValidationStatus.UNKNOWN,
            reason=VariationalValidationReason.VARIATIONAL_SOLVE_EXCEPTION,
            detail=f"variational solve raised {type(exc).__name__}: {exc}",
            solved_state=solved_state,
            domain=domain_evidence,
        )
    if not isinstance(solution, RaoSolution):
        return _base_outcome(
            status=VariationalValidationStatus.UNKNOWN,
            reason=VariationalValidationReason.VARIATIONAL_SOLVE_EXCEPTION,
            detail=(
                "variational solver returned "
                f"{type(solution).__name__}, not RaoSolution"
            ),
            solved_state=solved_state,
            domain=domain_evidence,
        )

    bvp_converged = bool(solution.control_surface.converged)
    if not bvp_converged:
        return _base_outcome(
            status=VariationalValidationStatus.UNKNOWN,
            reason=VariationalValidationReason.VARIATIONAL_NOT_CONVERGED,
            detail="variational BVP residual solve did not converge",
            solved_state=solved_state,
            solution=solution,
            domain=domain_evidence,
        )
    moc_closed = bool(solution.converged)
    moc_ineligibility = None if moc_closed else _moc_ineligibility_reason(solution)
    if not moc_closed and not _moc_smoothness_only_failure(solution):
        return _base_outcome(
            status=VariationalValidationStatus.UNKNOWN,
            reason=VariationalValidationReason.MOC_OR_PHYSICAL_CLOSURE_FAILED,
            detail=(
                f"{moc_ineligibility}; a discrepancy report is refused because "
                "the failed gate is not limited to forward-net smoothness"
            ),
            solved_state=solved_state,
            solution=solution,
            domain=domain_evidence,
        )
    thrust_diag = solution.construction_diagnostics.get("thrust_sanity", {})
    full_cde = (
        thrust_diag.get("gate_basis") == DIRECT_FULL_CDE_CF_BASIS
        and thrust_diag.get("cde_reconstruction_complete") is True
        and thrust_diag.get("surface_scope") == "full_control_surface_cde"
    )
    try:
        cf_variational = _host_float(
            thrust_diag.get("cf_surface"),
            "direct full-CDE thrust coefficient",
            positive=True,
        )
        solution_cf = _host_float(
            solution.thrust_coefficient,
            "RaoSolution.thrust_coefficient",
            positive=True,
        )
        if not math.isclose(
            cf_variational, solution_cf, rel_tol=1.0e-12, abs_tol=1.0e-12
        ):
            raise ValueError(
                "RaoSolution thrust coefficient does not match the recorded "
                "direct full-CDE surface integral"
            )
    except (TypeError, ValueError) as exc:
        cf_variational = None
        full_cde_detail = f"{type(exc).__name__}: {exc}"
    else:
        full_cde_detail = None
    if not full_cde or cf_variational is None:
        return _base_outcome(
            status=VariationalValidationStatus.UNKNOWN,
            reason=VariationalValidationReason.FULL_CDE_THRUST_UNAVAILABLE,
            detail=(
                "direct full C-D-E surface-integral thrust coefficient is "
                "unavailable; D-E-only or mass-scaled surrogates are refused"
                + (" (" + full_cde_detail + ")" if full_cde_detail else "")
            ),
            solved_state=solved_state,
            solution=solution,
            domain=domain_evidence,
        )

    # Defense in depth: B5's map already required valid Eq. (5) samples, but
    # the independently solved candidate must establish the same result after
    # its numerical and physical closure gates have passed.
    validity = solution.rao_validity
    if validity is None or not validity.available:
        return _base_outcome(
            status=VariationalValidationStatus.UNKNOWN,
            reason=VariationalValidationReason.RAO_VALIDITY_UNAVAILABLE,
            detail=(
                "Rao 1999 Eq. (5) validity is absent or unevaluable; contour "
                "comparison is not admissible"
            ),
            solved_state=solved_state,
            solution=solution,
            domain=domain_evidence,
        )
    if not validity.valid:
        if not moc_closed:
            return _base_outcome(
                status=VariationalValidationStatus.UNKNOWN,
                reason=VariationalValidationReason.MOC_OR_PHYSICAL_CLOSURE_FAILED,
                detail=(
                    f"{moc_ineligibility}; the negative Eq. (5) result from "
                    "this non-closed MOC attempt is diagnostic, not an "
                    "adjudication"
                ),
                solved_state=solved_state,
                solution=solution,
                domain=domain_evidence,
            )
        return _base_outcome(
            status=VariationalValidationStatus.FAIL,
            reason=VariationalValidationReason.INVALID_RAO_REGION,
            detail=(
                "the converged variational solution's Rao 1999 Eq. (5) "
                "validity predicts a negative radial increment"
            ),
            solved_state=solved_state,
            solution=solution,
            domain=domain_evidence,
            available=True,
        )
    try:
        contour = solution.to_contour_dict(
            Rt=Rt,
            epsilon=epsilon,
            length_pct=length_pct,
            pa_over_p0=ambient / Pc,
            Ru_factor=ru_factor,
            Rd_factor=rd_factor,
            convergent_half_angle_deg=convergent_angle,
        )
    except RaoInvalidRegionError as exc:
        # Validity was checked above; an export refusal here is still explicit
        # unknown evidence, never a request to reuse the TOP coordinates.
        return _base_outcome(
            status=VariationalValidationStatus.UNKNOWN,
            reason=VariationalValidationReason.CONTOUR_EXPORT_REFUSED,
            detail=str(exc),
            solved_state=solved_state,
            solution=solution,
            domain=domain_evidence,
        )
    except (ArithmeticError, IndexError, KeyError, TypeError, ValueError) as exc:
        return _base_outcome(
            status=VariationalValidationStatus.UNKNOWN,
            reason=VariationalValidationReason.NONFINITE_OR_MALFORMED_GEOMETRY,
            detail=(
                "variational contour export is malformed or unavailable: "
                f"{type(exc).__name__}: {exc}"
            ),
            solved_state=solved_state,
            solution=solution,
            domain=domain_evidence,
        )
    try:
        discrepancy = compare_top_and_variational_contours(
            analytic_x=analytic_x,
            analytic_r=analytic_r,
            analytic_throat_index=throat_index,
            variational_x=contour["x"],
            variational_r=contour["y"],
            Rt=Rt,
            analytic_thrust_coefficient=cf_top,
            variational_thrust_coefficient=cf_variational,
            tolerances=config.tolerances,
            comparison_points=config.comparison_points,
            adjudication_eligible=moc_closed,
            adjudication_ineligibility_reason=moc_ineligibility,
        )
    except (KeyError, TypeError, ValueError) as exc:
        reason = (
            VariationalValidationReason.NONFINITE_DISCREPANCY
            if "discrep" in str(exc).lower()
            else VariationalValidationReason.NONFINITE_OR_MALFORMED_GEOMETRY
        )
        return _base_outcome(
            status=VariationalValidationStatus.UNKNOWN,
            reason=reason,
            detail=f"contour comparison unavailable: {type(exc).__name__}: {exc}",
            solved_state=solved_state,
            solution=solution,
            domain=domain_evidence,
        )

    if not moc_closed:
        return _base_outcome(
            status=VariationalValidationStatus.UNKNOWN,
            reason=VariationalValidationReason.MOC_OR_PHYSICAL_CLOSURE_FAILED,
            detail=(
                f"{moc_ineligibility}; geometry/performance discrepancies are "
                "reported for diagnosis but are not adjudication-eligible"
            ),
            solved_state=solved_state,
            solution=solution,
            domain=domain_evidence,
            discrepancy=discrepancy,
        )

    if not discrepancy.passes:
        return _base_outcome(
            status=VariationalValidationStatus.FAIL,
            reason=VariationalValidationReason.DISCREPANCY_EXCEEDED,
            detail=(
                "optimized TOP and fully solved variational contours exceed "
                "one or more stated geometry/performance tolerances"
            ),
            solved_state=solved_state,
            solution=solution,
            domain=domain_evidence,
            discrepancy=discrepancy,
            available=True,
        )
    return _base_outcome(
        status=VariationalValidationStatus.PASS,
        reason=VariationalValidationReason.WITHIN_TOLERANCES,
        detail=(
            "optimized TOP and fully solved variational contours agree within "
            "all stated geometry/performance tolerances"
        ),
        solved_state=solved_state,
        solution=solution,
        domain=domain_evidence,
        discrepancy=discrepancy,
        available=True,
    )


__all__ = [
    "ContourDiscrepancy",
    "ContourDiscrepancyTolerances",
    "CONTOUR_COMPARISON_POLICY_ID",
    "ANALYTIC_TOP_CF_BASIS",
    "DIRECT_FULL_CDE_CF_BASIS",
    "RaoConvergenceDomainLike",
    "RaoDomainEvidence",
    "VARIATIONAL_CONTOUR_MODEL_ID",
    "VARIATIONAL_FORMULATION",
    "VARIATIONAL_GAMMA_MODEL_ID",
    "VARIATIONAL_SOLVER_SOURCE_ID",
    "VARIATIONAL_VALIDATOR_MODEL_ID",
    "VARIATIONAL_VALIDATOR_SOURCE_ID",
    "VariationalContourValidationOutcome",
    "VariationalPostSolveConfig",
    "VariationalValidationReason",
    "VariationalValidationStatus",
    "compare_top_and_variational_contours",
    "unavailable_variational_validation",
    "validate_optimized_top_contour",
]
