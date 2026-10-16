"""
raosim.mdo.nlp — Phase 8/9: the ε-constraint hard-constrained NLP + Pareto sweep.

Objective/constraint policy (plan §8):

    min  m_objective      s.t.  I_sp ≥ I_sp,min          (the ε-constraint)
                                every *enforced* discipline margin ≥ 0

Both implemented objective identities use a smooth battery governing-branch
surrogate so gradients remain continuous. ``min_dry_mass_partial`` (the
default) adds liner, channel lands, and closeout to the electric-feed branch;
it is explicitly not full engine dry mass. Exact electric and resolved-partial
subtotals are reported separately.

sweeping ``I_sp,min`` traces the mass–performance frontier.  The engine is
always thrust-closed internally (the outer Newton), so thrust is NOT a free NLP
constraint — the design variables are the twelve hardware values of
``DesignVector`` (``D_cg`` pinned for a slotted pintle) and every
evaluation returns a converged, differentiable engine (``mdo.engine``).

Derivatives (plan §4.3, §12.2 row 8): a *low-dimensional, constraint-rich*
problem (n_x = 11–13, 28 hard rows), so total derivatives are exact through the two
IFT solves and handed to SLSQP with no finite-difference step noise.  The
constraint Jacobian is assembled by **reverse mode** (``jax.jacrev``) and the
scalar objective by ``jax.grad``: forward mode (``jacfwd``) through the jit'd
nested Optimistix root-finds drops the tangent of the geometry-only min-
aggregated margins (a forward-through-implicit-solve quirk), and at 28×13 reverse
mode is no costlier than forward here anyway.  Optimisation runs in the unit box
(``ScaledSpace``) so O(1e6) pressures and O(0.1) fractions are conditioned
together (Martins & Ning 2021 ch. 4).

**Coking is now enforced** (all of ``CONSTRAINT_NAMES``).  An earlier finding
showed the RP-1 liquid-wall coking limit (SP-8087, 728 K) was violated
everywhere in the box with pure regen and fixed channels — the wall is
coolant-enthalpy-limited, not film-coefficient-limited, so channel geometry
alone cannot satisfy it.  The fix (this commit) is **film cooling**: the design
vector now carries ``channel_width``, ``channel_height`` and a fuel ``film_frac``
that reduces the gas-side driving temperature over the chamber→throat region at
a delivered-c* penalty (SP-8087/SP-125 combined regen+film; §6.2b).  With that
lever coking is satisfiable, so the frontier is genuinely **thermal-limited**:
raising Isp wants less film (less c* penalty) but that pushes the wall toward
the coking limit — a real trade the optimiser now resolves.  ``enforced=`` can
still drop it for diagnostics.

Hard constraints only; penalties would be for feasibility restoration, not the
reported optimum (plan §8, §11 Phase-9 gate).

Robustness and verification (MDO issues audit 2026-10-16, §1.1 items 1-4)
--------------------------------------------------------------------------
* **A bad trial point is a rejected step, not an aborted solve.**  The
  compiled objective/constraint programs contain two implicit solves; a
  non-finite input reaching a Lineax linear solve raises an Equinox runtime
  error (``jax.errors.JaxRuntimeError``).  :class:`_GuardedProblem` turns a
  primal failure at a trial point into a finite *penalty* (objective above
  every finite value seen, every enforced row strongly violated), so SLSQP's
  merit line search backtracks.  A derivative failure at a point whose primal
  is finite falls back to central differences of the same compiled primal in
  the unit box.  The penalty is never a reported optimum: the returned point
  is re-evaluated from scratch, and a failure there yields ``success=False``
  with ``numerical_status="fail"``.  Configuration errors still raise.
* **Independent KKT check** (:func:`kkt_check`): stationarity, dual
  feasibility and complementarity recomputed at the returned point from the
  exact gradient and Jacobian, with multipliers from non-negative least
  squares, against the plan §11 stationarity tolerance (1e-4).  The SciPy
  ``success`` flag is no longer the only evidence of a converged optimum.
* **Multi-start** (``n_starts``): the reference start plus scrambled Sobol
  points in the unit box, compiled once and reused, with a basin count so an
  infeasible requirement set and a failed nonconvex solve can be told apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

import numpy as np

import raosim.jax  # noqa: F401  -- float64
import equinox as eqx
import jax
import jax.numpy as jnp
from scipy.optimize import OptimizeResult, minimize, nnls
from scipy.stats import qmc

from raosim.mdo.schema import (
    DesignVector, MissionSpec, VariableSpec, default_design_space,
)
from raosim.mdo.scaling import ScaledSpace
from raosim.mdo.engine import chamber_surfaces_for, solve_engine
from raosim.mdo.grid import GridTopology
from raosim.mdo.constraints import (
    ENGINE_CONSTRAINT_SPECS,
    OPTIMIZER_CONSTRAINT_NAMES,
    OPTIMIZER_CONSTRAINT_SCALES,
    OPTIMIZER_CONSTRAINT_SPECS,
    constraint_metadata,
    status_from_rows,
)
from raosim.mdo.nozzle import ContourProviderSelection
from raosim.mdo.coolant_htd import ModelCoverageError, require_htd_coverage
from raosim.mdo.objectives import (
    DEFAULT_MASS_OBJECTIVE,
    MassObjective,
    coerce_mass_objective,
    mass_objective_value,
)
from raosim.mdo.propellants import get_propellant

Array = jnp.ndarray

#: Generated from the single ordered manifest in ``mdo.constraints``.
CONSTRAINT_NAMES = OPTIMIZER_CONSTRAINT_NAMES
#: Every differentiable hard row from the manifest.  Applicability-specific
#: rows return an inert positive value when not applicable; requirements use
#: ``applicable_optimizer_names`` to record their exact enforced subset.
DEFAULT_ENFORCED = CONSTRAINT_NAMES

# per-constraint reference scales (bring each margin to ~O(1) for the QP).
# The three requirement screens arrive already dimensionless -- they are
# FRACTIONAL margins (1 - value/limit), which is O(1) at every thrust class by
# construction -- so their scale is exactly 1.  See
# raosim.mdo.envelope.fractional_margin for why the absolute form was rejected.
_C_SCALE = OPTIMIZER_CONSTRAINT_SCALES
_MASS_REF = 50.0  # kg, objective conditioning


@dataclass(frozen=True)
class _OptimizationSpace:
    """Reduced optimizer coordinates and their full physical reconstruction.

    A fixed design value is substituted before every physics evaluation; it
    is never represented by a zero-width affine scaling or a soft constraint.
    The engine and persisted results retain the established 12/13-coordinate
    physical layout, regardless of how many coordinates the caller frees.
    """

    physical_specs: tuple[VariableSpec, ...]
    free_specs: tuple[VariableSpec, ...]
    fixed_values: tuple[tuple[str, float], ...]

    @classmethod
    def for_mission(
        cls,
        mission: MissionSpec,
        fixed_variables: Mapping[str, float] | None = None,
        bounds: Mapping[str, tuple[float, float]] | None = None,
    ) -> "_OptimizationSpace":
        physical_specs = default_design_space(mission)
        physical_names = {spec.name for spec in physical_specs}
        fixed = dict(fixed_variables or {})
        requested_bounds = dict(bounds or {})
        unknown = (set(fixed) | set(requested_bounds)) - set(DesignVector.names())
        if unknown:
            raise ValueError(f"unknown design-variable policies: {sorted(unknown)}")

        # A mission with no active O/F lever owns its physical mixture ratio.
        # Accept an identical explicit pin, but never silently replace it or
        # turn flat thermochemistry into an artificial optimization lever.
        if "OF" not in physical_names:
            if "OF" in requested_bounds:
                raise ValueError("OF bounds require an active O/F-dependent model")
            if "OF" in fixed:
                if isinstance(fixed["OF"], (bool, np.bool_)):
                    raise ValueError("fixed OF must be a number, not bool")
                value = float(fixed.pop("OF"))
                if not np.isfinite(value) or value != float(mission.OF):
                    raise ValueError("fixed OF must agree with the mission's fixed OF")

        # The centre-gap diameter belongs to the movable pintle valve
        # (mdo/pintle.py).  A slotted post has no centre gap, so the variable
        # is pinned at its reference instead of becoming a dead optimizer
        # column (plan rule 4: architecture stays an outer choice).
        if getattr(mission, "pintle_architecture", "fixed_slotted") != (
            "son_movable"
        ):
            if "D_cg" in requested_bounds:
                raise ValueError(
                    "D_cg bounds require the son_movable pintle architecture"
                )
            fixed.setdefault("D_cg", next(
                spec.ref() for spec in physical_specs if spec.name == "D_cg"
            ))

        overlap = set(fixed) & set(requested_bounds)
        if overlap:
            raise ValueError(
                f"variables cannot be both fixed and bounded: {sorted(overlap)}"
            )
        for name, raw in fixed.items():
            if isinstance(raw, (bool, np.bool_)):
                raise ValueError(f"fixed {name} must be a number, not bool")
            value = float(raw)
            if not np.isfinite(value):
                raise ValueError(f"fixed {name} must be finite")
            if name == "film_frac":
                valid = 0.0 <= value < 1.0
            elif name == "eps":
                valid = value > 1.0
            else:
                valid = value > 0.0
            if not valid:
                raise ValueError(f"fixed {name} is outside its physical range")
            fixed[name] = value

        free_specs = []
        for spec in physical_specs:
            if spec.name in fixed:
                continue
            lo, hi = float(spec.lower), float(spec.upper)
            if spec.name in requested_bounds:
                try:
                    requested_lo, requested_hi = requested_bounds[spec.name]
                    if any(isinstance(value, (bool, np.bool_)) for value in (requested_lo, requested_hi)):
                        raise ValueError("boolean bound")
                    requested_lo = float(requested_lo)
                    requested_hi = float(requested_hi)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        f"{spec.name} bounds must be a pair of finite numbers"
                    ) from exc
                if (
                    not np.isfinite([requested_lo, requested_hi]).all()
                    or requested_lo >= requested_hi
                ):
                    raise ValueError(
                        f"{spec.name} bounds must be finite and strictly increasing"
                    )
                # Resolved jobs already perform this intersection and report
                # it. Apply the same contract to direct solve_min_mass calls.
                lo, hi = max(lo, requested_lo), min(hi, requested_hi)
            if not lo < hi:
                raise ValueError(
                    f"{spec.name} bounds do not intersect the mission search space"
                )
            free_specs.append(VariableSpec(
                spec.name, lo, hi, min(max(float(spec.ref()), lo), hi)
            ))
        return cls(physical_specs, tuple(free_specs), tuple(fixed.items()))

    @property
    def free_indices(self) -> tuple[int, ...]:
        names = tuple(spec.name for spec in self.physical_specs)
        return tuple(names.index(spec.name) for spec in self.free_specs)

    def reconstruct(self, free_values: Array) -> Array:
        """Rebuild the full physical vector without differentiating pins."""

        fixed = dict(self.fixed_values)
        values = jnp.asarray([
            fixed.get(spec.name, spec.ref()) for spec in self.physical_specs
        ], dtype=jnp.float64)
        indices = jnp.asarray(self.free_indices, dtype=int)
        return values.at[indices].set(free_values)

    def select_free(self, physical_values: Array) -> Array:
        return jnp.asarray(physical_values)[jnp.asarray(self.free_indices, dtype=int)]


def _constraint_vector(r, isp_min: float) -> Array:
    """Full scaled inequality vector g(x) ≥ 0 from a solved EngineResult."""
    raw = jnp.stack([
        (r.Isp - isp_min) if spec.engine_key is None
        else r.constraints[spec.engine_key]
        for spec in OPTIMIZER_CONSTRAINT_SPECS
    ])
    return raw / jnp.asarray(_C_SCALE)


def _raw_margins(r, isp_min: float) -> dict:
    margins = {
        spec.name: float(
            r.Isp - isp_min if spec.engine_key is None
            else r.constraints[spec.engine_key]
        )
        for spec in OPTIMIZER_CONSTRAINT_SPECS
    }
    # Final gates are mandatory post-solve checks rather than SLSQP rows.  Keep
    # every one in the result so an optimizer-successful but physically or
    # numerically invalid candidate cannot disappear behind the hard-row
    # summary.  Report-only diagnostics (currently thrust closure) remain
    # outside this margin mapping because they are not feasibility rows.
    for spec in ENGINE_CONSTRAINT_SPECS:
        if spec.optimizer_role == "final_gate":
            margins[spec.name] = float(r.constraints[spec.engine_key])
    return margins


def _margin_names() -> tuple[str, ...]:
    """Keys of :func:`_raw_margins`, in the same order, without an engine."""

    return tuple(spec.name for spec in OPTIMIZER_CONSTRAINT_SPECS) + tuple(
        spec.name for spec in ENGINE_CONSTRAINT_SPECS
        if spec.optimizer_role == "final_gate"
    )


# --------------------------------------------------------------------------- #
# Numerical policy constants (repository choices, stated once)                #
# --------------------------------------------------------------------------- #
#: Runtime failures of the compiled programs at one trial point.  Equinox's
#: ``error_if`` (used by Lineax/Optimistix when a linear solve sees non-finite
#: input) surfaces as ``jax.errors.JaxRuntimeError`` under ``jax.jit`` and as
#: ``equinox.EquinoxRuntimeError`` when the same code runs un-jitted, e.g. the
#: final re-evaluation below.  ``ValueError``/``TypeError`` (trace-time
#: configuration errors) and :class:`ModelCoverageError` are deliberately NOT
#: in this tuple: they describe the request, not the trial point, and still
#: raise.
_EVALUATION_FAILURES: tuple[type[BaseException], ...] = (
    jax.errors.JaxRuntimeError,
    eqx.EquinoxRuntimeError,
    FloatingPointError,
)
#: Penalty returned for a failed primal: the objective is placed this many
#: (|reference|-scaled) units above the LARGEST finite objective seen so far,
#: and every enforced scaled row is set to this violated value.  The largest
#: rather than the smallest finite value is used so the penalty also exceeds
#: the merit of the current (possibly worse than best) iterate.  Both numbers
#: only need to make SLSQP's L1 merit function larger than at the accepted
#: iterate so its line search backtracks; they are never a reported result.
_PENALTY_OBJECTIVE_OFFSET = 10.0
_PENALTY_CONSTRAINT = -10.0
#: Central-difference step in the unit box for the derivative fallback (kept
#: inside [0, 1]; one-sided at a bound).  The implicit inner roots are solved
#: to 1e-12 relative, so a 1e-6 step keeps roundoff near 1e-6 relative -
#: adequate for a fallback that is only taken when the exact IFT derivative
#: is unavailable.  tests/test_mdo_nlp.py documents that 1e-4 sits on the
#: plateau for the constraint rows; the fallback is not a verification tool.
_FD_STEP = 1.0e-6
#: Active-set tolerance on a SCALED margin.  Equal to the optimizer
#: feasibility tolerance behind ``optimizer_status`` (1e-5), so "feasible"
#: and "active" use one threshold.
_KKT_ACTIVE_TOL = 1.0e-5
#: A unit-box coordinate this close to 0 or 1 is on its bound.  SLSQP does
#: not always land exactly on a face: the Son-valve optimum leaves t_post at
#: u = 1.5e-9, and a 1e-9 tolerance then dropped that bound from the active
#: set and reported a false KKT failure (stationarity 2e-3 instead of 2e-6).
#: Including a near-face coordinate cannot hide non-stationarity, because NNLS
#: gives it a zero multiplier when its gradient does not need one, and any
#: multiplier it does take is charged in the complementarity report.
_KKT_BOUND_TOL = 1.0e-6
#: Plan §11 numerical definition of done: "KKT residual <= about 1e-4".
_KKT_STATIONARITY_TOL = 1.0e-4
#: Two converged feasible starts are in one basin when their unit-box points
#: agree to this max-norm distance AND their objectives to this relative
#: tolerance.  Repository choices; no held source states a multi-start
#: agreement criterion (the plan gives only the qualitative test).
_BASIN_DISTANCE_TOL = 1.0e-3
_BASIN_OBJECTIVE_RTOL = 1.0e-6


@dataclass(frozen=True)
class NLPResult:
    success: bool
    isp_min: float
    x: np.ndarray                 # full physical mission vector (12/13), incl pins
    design: dict                  # fixed 13-value physical contract
    objective_name: str
    objective_mass: float
    electric_package_objective_mass: float
    dry_mass_partial_objective_mass: float
    exact_electric_package_mass: float
    exact_dry_mass_partial: float
    Isp: float
    constraints: dict             # named RAW margins (≥0 feasible), ALL reported
    enforced: tuple               # which were hard constraints
    max_violation: float          # max scaled violation over the ENFORCED set
    optimizer_status: str         # pass/fail for the selected hard rows
    numerical_status: str         # pass/fail for inner roots and finiteness
    physics_status: str           # pass/fail/unknown physical-model verdict
    requirements_status: str      # pass/fail/unknown requirement-row verdict
    workflow_status: str          # pass/fail for the outer optimizer execution
    optimizer_feasible: bool      # compatibility: optimizer_status == pass
    feasible: bool                # compatibility: True only for a complete pass
    n_iter: int
    message: str
    # --- fail-closed evaluation (totals over every start of this solve) ---- #
    #: Trial points whose compiled primal raised or returned a non-finite
    #: value; each was answered with the backtracking penalty.
    evaluation_failures: int = 0
    #: Derivative requests answered by the central-difference fallback
    #: because the exact (IFT) derivative raised or was non-finite.
    derivative_fallbacks: int = 0
    #: Derivative requests that neither the exact derivative nor the fallback
    #: could answer.  The run that hit one ends as a typed failure.
    derivative_failures: int = 0
    # --- independent KKT check at the returned point (kkt_check) ----------- #
    #: pass / fail / unknown.  ``unknown`` when no derivative is available or
    #: nothing was optimized.  Does not alter ``optimizer_status``.
    kkt_status: str = "unknown"
    #: ||∇f − Jᵀλ − μ_lo + μ_hi||_∞ / max(1, ||∇f||_∞), scaled unit-box space.
    kkt_stationarity: float = float("nan")
    #: max |λ_i g_i|, |μ_lo,j u_j|, |μ_hi,j (1 − u_j)| over the active set.
    kkt_complementarity: float = float("nan")
    #: smallest multiplier (≥ 0 by construction of the NNLS fit).
    kkt_dual_feasibility: float = float("nan")
    #: enforced rows with scaled margin ≤ 1e-5, plus ``lower:``/``upper:``
    #: bound-active free variables.
    active_constraints: tuple = ()
    #: multipliers by enforced-row name (zero when inactive) plus active-bound
    #: entries, in scaled objective / scaled-margin units.
    multipliers: Mapping[str, float] = field(default_factory=dict)
    kkt_message: str = ""
    # --- multi-start ------------------------------------------------------- #
    n_starts: int = 1
    selected_start: int = 0
    #: distinct converged feasible basins found over all starts.
    n_basins: int = 0
    #: one summary dict per start (statuses, objective, unit-box start/end).
    start_summaries: tuple = ()

    @property
    def package_mass(self) -> float:
        """Deprecated alias for the smooth optimization objective."""

        return self.objective_mass


# --------------------------------------------------------------------------- #
# Independent KKT check (R3)                                                  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class KKTReport:
    """First-order optimality evidence at one point of the scaled problem.

    Problem (the exact space the optimizer used)::

        min f(u)  s.t.  g_i(u) ≥ 0 (enforced scaled rows),  0 ≤ u ≤ 1

    Lagrangian ``L = f − Σλ_i g_i − Σμ_lo,j u_j − Σμ_hi,j (1 − u_j)``, so
    stationarity is ``∇f − Jᵀλ − μ_lo + μ_hi = 0``.  These are the standard
    textbook first-order conditions; no held source in the corpus states the
    general form or a tolerance, so the 1e-4 threshold is the plan §11
    repository definition of done.
    """

    status: str
    stationarity: float
    complementarity: float
    dual_feasibility: float
    primal_violation: float
    active_constraints: tuple[str, ...]
    multipliers: dict
    rank: int
    message: str


def _unknown_kkt(message: str) -> KKTReport:
    nan = float("nan")
    return KKTReport("unknown", nan, nan, nan, nan, (), {}, 0, message)


def kkt_check(
    grad_f: Any,
    jac_g: Any,
    g: Any,
    u: Any,
    *,
    constraint_names: tuple[str, ...],
    variable_names: tuple[str, ...],
    active_tol: float = _KKT_ACTIVE_TOL,
    bound_tol: float = _KKT_BOUND_TOL,
    stationarity_tol: float = _KKT_STATIONARITY_TOL,
) -> KKTReport:
    """Recompute KKT stationarity, dual feasibility and complementarity.

    The active set is every row with ``g_i ≤ active_tol`` plus every
    coordinate within ``bound_tol`` of a box face.  Inactive rows and free
    coordinates get zero multipliers, which enforces complementarity by
    construction.  The active multipliers solve

        w = argmin_{w ≥ 0} || B w − ∇f ||₂,
        B = [ J_A ᵀ | +e_j (lower-active) | −e_j (upper-active) ]

    by non-negative least squares (``scipy.optimize.nnls``), so dual
    feasibility also holds by construction and the remaining residual is the
    honest stationarity error.  When LICQ fails (duplicate active gradients)
    the multipliers are not unique but the residual still is.

    ``status`` is ``pass`` when the normalized stationarity residual is at
    most ``stationarity_tol`` AND the point is primal feasible to
    ``active_tol`` (a violated point is not a KKT point, whatever its
    residual), ``fail`` otherwise, and ``unknown`` when any input is
    non-finite or there is no free coordinate.
    """

    grad_f = np.asarray(grad_f, dtype=float).reshape(-1)
    n = grad_f.size
    u = np.asarray(u, dtype=float).reshape(-1)
    g = np.asarray(g, dtype=float).reshape(-1)
    m = g.size
    jac = np.asarray(jac_g, dtype=float).reshape(m, n) if m else np.zeros((0, n))
    if len(constraint_names) != m or len(variable_names) != n or u.size != n:
        raise ValueError("kkt_check: names and array shapes disagree")
    if n == 0:
        return _unknown_kkt("no free design variable; nothing was optimized")
    if not (np.isfinite(grad_f).all() and np.isfinite(jac).all()
            and np.isfinite(g).all() and np.isfinite(u).all()):
        return _unknown_kkt("non-finite gradient, Jacobian or margin")

    active_rows = np.flatnonzero(g <= active_tol)
    lower = np.flatnonzero(u <= bound_tol)
    upper = np.flatnonzero(u >= 1.0 - bound_tol)
    identity = np.eye(n)
    columns = (
        [jac[i] for i in active_rows]
        + [identity[j] for j in lower]
        + [-identity[j] for j in upper]
    )
    if columns:
        basis = np.column_stack(columns)
        weights, _ = nnls(basis, grad_f)
        residual = grad_f - basis @ weights
        rank = int(np.linalg.matrix_rank(basis))
    else:
        weights = np.zeros(0)
        residual = grad_f.copy()
        rank = 0
    k, nl = active_rows.size, lower.size
    lam = np.zeros(m)
    lam[active_rows] = weights[:k]
    mu_lo = np.zeros(n)
    mu_lo[lower] = weights[k:k + nl]
    mu_hi = np.zeros(n)
    mu_hi[upper] = weights[k + nl:]

    stationarity = float(
        np.max(np.abs(residual)) / max(1.0, float(np.max(np.abs(grad_f))))
    )
    complementarity = float(max(
        np.max(np.abs(lam * g), initial=0.0),
        np.max(np.abs(mu_lo * u), initial=0.0),
        np.max(np.abs(mu_hi * (1.0 - u)), initial=0.0),
    ))
    dual = float(np.min(weights)) if weights.size else 0.0
    primal_violation = float(max(
        0.0, -np.min(g, initial=0.0), -np.min(u), np.max(u) - 1.0,
    ))
    active = tuple(constraint_names[i] for i in active_rows) + tuple(
        f"lower:{variable_names[j]}" for j in lower
    ) + tuple(f"upper:{variable_names[j]}" for j in upper)
    multipliers = {
        name: float(lam[i]) for i, name in enumerate(constraint_names)
    }
    multipliers.update({
        f"lower:{variable_names[j]}": float(mu_lo[j]) for j in lower
    })
    multipliers.update({
        f"upper:{variable_names[j]}": float(mu_hi[j]) for j in upper
    })
    feasible = primal_violation <= active_tol
    status = "pass" if (stationarity <= stationarity_tol and feasible) else "fail"
    message = (
        f"stationarity {stationarity:.2e} (tol {stationarity_tol:.0e}), "
        f"complementarity {complementarity:.2e}, "
        f"{len(active)} active of {m} rows + {2 * n} bounds, rank {rank}"
    )
    if not feasible:
        message += f"; primal violation {primal_violation:.2e} > {active_tol:.0e}"
    return KKTReport(
        status=status,
        stationarity=stationarity,
        complementarity=complementarity,
        dual_feasibility=dual,
        primal_violation=primal_violation,
        active_constraints=active,
        multipliers=multipliers,
        rank=rank,
        message=message,
    )


# --------------------------------------------------------------------------- #
# Fail-closed trial-point evaluation (R1)                                     #
# --------------------------------------------------------------------------- #
class _DerivativeUnavailable(RuntimeError):
    """No exact or finite-difference derivative exists at an iterate.

    Raised out of the SLSQP callbacks so the run stops instead of receiving a
    zero-filled gradient, which SLSQP would read as a flat (converged) point.
    """

    def __init__(self, u: np.ndarray, what: str):
        super().__init__(f"{what} unavailable at an iterate")
        self.u = np.asarray(u, dtype=float).copy()


def _describe_failure(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    first = next((line for line in text if line.strip()), "")
    return f"{type(exc).__name__}: {first[:200]}"


class _GuardedProblem:
    """SLSQP-facing wrapper that never lets a trial-point failure escape.

    One instance serves one optimizer run.  Primal values are evaluated
    jointly (objective and enforced rows at the same point) and cached by the
    exact unit-box coordinates, so SLSQP's separate ``fun``/``constraints``
    calls see one consistent verdict per point.
    """

    def __init__(self, obj, obj_grad, con, con_jac, n_constraints: int):
        self._obj = obj
        self._obj_grad = obj_grad
        self._con = con
        self._con_jac = con_jac
        self.n_constraints = int(n_constraints)
        self.evaluation_failures = 0
        self.derivative_fallbacks = 0
        self.derivative_failures = 0
        self.max_objective: float | None = None
        self.last_failure: str = ""
        self._primal_cache: dict[bytes, tuple[float, np.ndarray, bool]] = {}

    # -- primal -------------------------------------------------------------- #
    def primal(self, u: np.ndarray) -> tuple[float, np.ndarray, bool]:
        u = np.asarray(u, dtype=float)
        key = u.tobytes()
        hit = self._primal_cache.get(key)
        if hit is not None:
            return hit
        ok = True
        try:
            f = float(self._obj(jnp.asarray(u)))
            c = np.asarray(self._con(jnp.asarray(u)), dtype=float).reshape(-1)
        except _EVALUATION_FAILURES as exc:
            ok = False
            self.last_failure = _describe_failure(exc)
            f, c = float("nan"), np.full(self.n_constraints, np.nan)
        if ok and not (np.isfinite(f) and np.isfinite(c).all()):
            ok = False
            self.last_failure = "non-finite objective or constraint value"
        if ok:
            if self.max_objective is None or f > self.max_objective:
                self.max_objective = f
        else:
            self.evaluation_failures += 1
        entry = (f, c, ok)
        if len(self._primal_cache) > 512:
            self._primal_cache.clear()
        self._primal_cache[key] = entry
        return entry

    def _penalty(self) -> tuple[float, np.ndarray]:
        ref = 0.0 if self.max_objective is None else self.max_objective
        f = ref + _PENALTY_OBJECTIVE_OFFSET * max(1.0, abs(ref))
        return f, np.full(self.n_constraints, _PENALTY_CONSTRAINT)

    def fun(self, u: np.ndarray) -> float:
        f, _, ok = self.primal(u)
        return f if ok else self._penalty()[0]

    def cons(self, u: np.ndarray) -> np.ndarray:
        _, c, ok = self.primal(u)
        return c if ok else self._penalty()[1]

    # -- derivatives --------------------------------------------------------- #
    def gradient(self, u: np.ndarray) -> tuple[np.ndarray, bool]:
        """(∇f, available).  Exact reverse mode, else central differences."""
        u = np.asarray(u, dtype=float)
        try:
            value = np.asarray(self._obj_grad(jnp.asarray(u)), dtype=float)
            if np.isfinite(value).all():
                return value.reshape(u.size), True
        except _EVALUATION_FAILURES as exc:
            self.last_failure = _describe_failure(exc)
        return self._finite_difference(u, objective=True)

    def jacobian(self, u: np.ndarray) -> tuple[np.ndarray, bool]:
        u = np.asarray(u, dtype=float)
        try:
            value = np.asarray(self._con_jac(jnp.asarray(u)), dtype=float)
            if np.isfinite(value).all():
                return value.reshape(self.n_constraints, u.size), True
        except _EVALUATION_FAILURES as exc:
            self.last_failure = _describe_failure(exc)
        return self._finite_difference(u, objective=False)

    def grad(self, u: np.ndarray) -> np.ndarray:
        value, complete = self.gradient(u)
        if not complete:
            raise _DerivativeUnavailable(u, "objective gradient")
        return value

    def jac(self, u: np.ndarray) -> np.ndarray:
        value, complete = self.jacobian(u)
        if not complete:
            raise _DerivativeUnavailable(u, "constraint Jacobian")
        return value

    def _finite_difference(
        self, u: np.ndarray, *, objective: bool
    ) -> tuple[np.ndarray, bool]:
        n = u.size
        shape = (n,) if objective else (self.n_constraints, n)
        f0, c0, ok0 = self.primal(u)
        if not ok0:
            # No finite primal here, so there is no derivative to recover.
            # The caller (grad/jac) turns this into _DerivativeUnavailable.
            self.derivative_failures += 1
            return np.zeros(shape), False
        self.derivative_fallbacks += 1
        v0 = f0 if objective else c0
        out = np.zeros(shape)
        complete = True
        for j in range(n):
            hi = min(u[j] + _FD_STEP, 1.0)
            lo = max(u[j] - _FD_STEP, 0.0)
            up, um = u.copy(), u.copy()
            up[j], um[j] = hi, lo
            fp, cp, okp = self.primal(up)
            fm, cm, okm = self.primal(um)
            vp = fp if objective else cp
            vm = fm if objective else cm
            if okp and okm and hi > lo:
                column = (vp - vm) / (hi - lo)
            elif okp and hi > u[j]:
                column = (vp - v0) / (hi - u[j])
            elif okm and u[j] > lo:
                column = (v0 - vm) / (u[j] - lo)
            else:
                column = np.zeros(shape[:-1]) if not objective else 0.0
                complete = False
            if objective:
                out[j] = column
            else:
                out[:, j] = column
        if not complete:
            self.derivative_failures += 1
        return out, complete


# --------------------------------------------------------------------------- #
# Compiled problem                                                            #
# --------------------------------------------------------------------------- #
def _make_callables(
    mission,
    isp_min,
    couple,
    enforced_idx,
    objective=DEFAULT_MASS_OBJECTIVE,
    surfaces=None,
    contour_provider: ContourProviderSelection = None,
    *,
    optimization_space: _OptimizationSpace | None = None,
    topology: GridTopology | None = None,
):
    objective = coerce_mass_objective(objective)
    if surfaces is None:
        # Backward-compatible internal/testing entry point.  Resolution still
        # performs the same schema, propellant-identity, and domain validation
        # as the public solve.
        surfaces = chamber_surfaces_for(mission)
    policy_space = (
        _OptimizationSpace.for_mission(mission)
        if optimization_space is None else optimization_space
    )
    layout = mission.design_layout()
    ss = ScaledSpace.from_specs(policy_space.free_specs)
    idx = jnp.asarray(enforced_idx, dtype=int)

    # A non-default station grid is forwarded only when requested, so callers
    # (and test doubles) that never pass one see the historical call.
    grid_kw = {} if topology is None else {"topo": topology}

    def solve(u):
        return solve_engine(
            DesignVector.from_active_array(
                policy_space.reconstruct(ss.to_physical(u)),
                layout, fixed_of=mission.OF
            ),
            mission,
            couple_eta_cstar=couple,
            surfaces=surfaces,
            contour_provider=contour_provider,
            **grid_kw,
        )

    @jax.jit
    def obj(u):
        return mass_objective_value(solve(u), objective) / _MASS_REF

    @jax.jit
    def con(u):
        return _constraint_vector(solve(u), isp_min)[idx]

    # Reverse-mode (jacrev) for the constraint Jacobian: forward-mode (jacfwd)
    # through the jit'd nested Optimistix root-finds drops the tangent of the
    # geometry-only min-aggregated margins (e.g. land_min) — a forward-through-
    # implicit-solve quirk; jacrev is exact (FD-verified). For this
    # low-dimensional, constraint-rich problem reverse mode remains appropriate.
    return ss, obj, jax.jit(jax.grad(obj)), con, jax.jit(jax.jacrev(con))


@dataclass(frozen=True)
class _SolveContext:
    """Everything one solve shares across its starts (compiled once)."""

    mission: MissionSpec
    isp_min: float
    enforced: tuple[str, ...]
    enforced_idx: tuple[int, ...]
    policy_space: _OptimizationSpace
    scaled_space: ScaledSpace
    surfaces: Any
    couple_eta_cstar: bool
    contour_provider: ContourProviderSelection
    objective: MassObjective
    obj: Any
    obj_grad: Any
    con: Any
    con_jac: Any
    method: str
    maxiter: int
    topology: GridTopology | None = None

    def guard(self) -> _GuardedProblem:
        return _GuardedProblem(
            self.obj, self.obj_grad, self.con, self.con_jac,
            len(self.enforced_idx),
        )

    def physical(self, u: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.policy_space.reconstruct(self.scaled_space.to_physical(
                jnp.asarray(u)
            ))
        )


def _contract_design(context: _SolveContext, x_phys: np.ndarray) -> dict:
    """Thirteen-value contract dict for a physical vector, without physics."""

    layout = context.mission.design_layout()
    vector = DesignVector.from_active_array(
        jnp.asarray(x_phys), layout, fixed_of=context.mission.OF
    )
    return {
        name: float(value)
        for name, value in vector.as_contract_dict().items()
    }


def _failure_result(
    *,
    isp_min: float,
    x_phys: np.ndarray,
    design: dict,
    objective_name: str,
    enforced: tuple[str, ...],
    n_iter: int,
    message: str,
    evaluation_failures: int = 0,
    derivative_fallbacks: int = 0,
    derivative_failures: int = 0,
) -> NLPResult:
    """Typed failed point: no finite engine evaluation backs this design."""

    nan = float("nan")
    return NLPResult(
        success=False,
        isp_min=float(isp_min),
        x=np.asarray(x_phys, dtype=float),
        design=dict(design),
        objective_name=objective_name,
        objective_mass=nan,
        electric_package_objective_mass=nan,
        dry_mass_partial_objective_mass=nan,
        exact_electric_package_mass=nan,
        exact_dry_mass_partial=nan,
        Isp=nan,
        constraints={name: nan for name in _margin_names()},
        enforced=tuple(enforced),
        max_violation=float("inf"),
        optimizer_status="fail",
        numerical_status="fail",
        physics_status="unknown",
        requirements_status="unknown",
        workflow_status="fail",
        optimizer_feasible=False,
        feasible=False,
        n_iter=int(n_iter),
        message=message,
        evaluation_failures=int(evaluation_failures),
        derivative_fallbacks=int(derivative_fallbacks),
        derivative_failures=int(derivative_failures),
        kkt_status="unknown",
        kkt_message="no finite evaluation at the returned point",
    )


def _optimize(context: _SolveContext, guard: _GuardedProblem,
              start: np.ndarray) -> OptimizeResult:
    space = context.policy_space.free_specs
    if not space:
        return OptimizeResult(
            x=start, success=True, nit=0,
            message="All design variables fixed; evaluated without optimization",
        )
    return minimize(
        guard.fun, start, jac=guard.grad, method=context.method,
        bounds=[(0.0, 1.0)] * len(space),
        constraints=(
            [{"type": "ineq", "fun": guard.cons, "jac": guard.jac}]
            if context.enforced_idx else []
        ),
        options={"maxiter": context.maxiter, "ftol": 1e-9},
    )


def _assemble_result(
    context: _SolveContext,
    res: OptimizeResult,
    guard: _GuardedProblem,
) -> NLPResult:
    """Re-evaluate the returned point from scratch and grade it."""

    mission = context.mission
    isp_min = context.isp_min
    enforced_idx = context.enforced_idx
    u = np.clip(np.asarray(res.x, dtype=float), 0.0, 1.0)
    x_phys = context.physical(u)
    layout = mission.design_layout()
    n_iter = int(getattr(res, "nit", -1))
    try:
        design_vector = DesignVector.from_active_array(
            jnp.asarray(x_phys), layout, fixed_of=mission.OF
        )
        r = solve_engine(
            design_vector,
            mission,
            couple_eta_cstar=context.couple_eta_cstar,
            surfaces=context.surfaces,
            contour_provider=context.contour_provider,
            **({} if context.topology is None else {"topo": context.topology}),
        )
        # Materialize everything while still inside the guard: JAX dispatch
        # is asynchronous, so a runtime error can surface on first read.
        scaled_all = np.asarray(_constraint_vector(r, isp_min), dtype=float)
        engine_values = np.asarray([
            float(r.constraints[spec.engine_key])
            for spec in ENGINE_CONSTRAINT_SPECS
        ])
        design = {
            name: float(value)
            for name, value in design_vector.as_contract_dict(
                effective_of=r.OF
            ).items()
        }
        objective_mass = float(mass_objective_value(r, context.objective))
        masses = (
            float(r.electric_package_objective_mass),
            float(r.dry_mass_partial_objective_mass),
            float(r.electric_package_exact_mass),
            float(r.dry_mass_partial_exact_mass),
        )
        isp = float(r.Isp)
        raw_margins = _raw_margins(r, isp_min)
    except _EVALUATION_FAILURES as exc:
        return _failure_result(
            isp_min=isp_min,
            x_phys=x_phys,
            design=_contract_design(context, x_phys),
            objective_name=context.objective.value,
            enforced=context.enforced,
            n_iter=n_iter,
            message=(
                "numerical evaluation failed at the returned point "
                f"({_describe_failure(exc)}); optimizer reported: "
                f"{res.message}"
            ),
            evaluation_failures=guard.evaluation_failures,
            derivative_fallbacks=guard.derivative_fallbacks,
            derivative_failures=guard.derivative_failures,
        )

    enforced_values = scaled_all[list(enforced_idx)]
    viol = (
        float(max(0.0, -np.min(enforced_values, initial=0.0)))
        if np.isfinite(enforced_values).all() else float("inf")
    )
    optimizer_status = "pass" if viol < 1e-5 else "fail"
    applicable, available, required, _ = constraint_metadata(
        mission, context.surfaces
    )
    numerical_indices = tuple(
        i
        for i, spec in enumerate(ENGINE_CONSTRAINT_SPECS)
        if spec.category == "numerical"
    )
    engine_physics_indices = tuple(
        i
        for i, spec in enumerate(ENGINE_CONSTRAINT_SPECS)
        if spec.category not in {"requirement", "numerical"}
    )
    numerical_status = status_from_rows(
        engine_values,
        applicable,
        available,
        required,
        indices=numerical_indices,
    )
    requirement_indices = tuple(
        i
        for i, spec in enumerate(OPTIMIZER_CONSTRAINT_SPECS)
        if spec.category == "requirement"
    )
    requirement_rows_finite = all(
        np.isfinite(scaled_all[i]) for i in requirement_indices
    )
    if numerical_status != "pass" or not requirement_rows_finite:
        requirements_status = "unknown"
    elif any(scaled_all[i] < -1.0e-5 for i in requirement_indices):
        requirements_status = "fail"
    else:
        requirements_status = "pass"
    # Preserve the optimizer's documented 1e-5 scaled feasibility tolerance
    # for hard rows: SLSQP routinely returns active constraints a few ulps
    # below zero.  Mandatory final gates receive no such relaxation; they are
    # post-solve validity checks and therefore fail on any demonstrated
    # negative margin.
    physics_values = engine_values.copy()
    for i, spec in enumerate(ENGINE_CONSTRAINT_SPECS):
        if (
            spec.optimizer_role == "hard"
            and spec.category not in {"requirement", "numerical"}
            and np.isfinite(physics_values[i])
            and -1.0e-5 * spec.scale <= physics_values[i] < 0.0
        ):
            physics_values[i] = 0.0
    physics_status = status_from_rows(
        physics_values,
        applicable,
        available,
        required,
        indices=engine_physics_indices,
        nonfinite="unknown",
    )
    if physics_status == "pass" and numerical_status != "pass":
        # A failed numerical evaluation cannot establish a physical pass.
        physics_status = "unknown"
    kkt = _kkt_at(context, guard, u)
    # SciPy's success flag is not an optimality certificate: a run that
    # stalls at the edge of a failing region, or stops on a zero step,
    # reports success.  The returned point is a solved optimum only when the
    # independent first-order check confirms it.  ``unknown`` counts as not
    # confirmed unless nothing was optimized (every variable fixed).
    kkt_confirms = kkt.status == "pass" or (
        kkt.status == "unknown" and not context.policy_space.free_specs
    )
    workflow_ok = (
        bool(res.success) and numerical_status == "pass" and kkt_confirms
    )
    workflow_status = "pass" if workflow_ok else "fail"
    optimizer_feasible = optimizer_status == "pass"
    complete_pass = (
        workflow_status == "pass"
        and optimizer_status == "pass"
        and numerical_status == "pass"
        and physics_status == "pass"
        and requirements_status == "pass"
    )
    message = str(res.message)
    if guard.evaluation_failures:
        message += (
            f" [{guard.evaluation_failures} trial point(s) failed numerically "
            f"and were rejected; last: {guard.last_failure}]"
        )
    if bool(res.success) and not kkt_confirms:
        message += f" [returned point is not a confirmed KKT point: {kkt.message}]"
    return NLPResult(
        success=workflow_ok,
        isp_min=float(isp_min),
        x=x_phys,
        design=design,
        objective_name=context.objective.value,
        objective_mass=objective_mass,
        electric_package_objective_mass=masses[0],
        dry_mass_partial_objective_mass=masses[1],
        exact_electric_package_mass=masses[2],
        exact_dry_mass_partial=masses[3],
        Isp=isp,
        constraints=raw_margins, enforced=tuple(context.enforced),
        max_violation=viol,
        optimizer_status=optimizer_status,
        numerical_status=numerical_status,
        requirements_status=requirements_status,
        optimizer_feasible=optimizer_feasible,
        physics_status=physics_status,
        workflow_status=workflow_status,
        feasible=complete_pass,
        n_iter=n_iter, message=message,
        evaluation_failures=guard.evaluation_failures,
        derivative_fallbacks=guard.derivative_fallbacks,
        derivative_failures=guard.derivative_failures,
        kkt_status=kkt.status,
        kkt_stationarity=kkt.stationarity,
        kkt_complementarity=kkt.complementarity,
        kkt_dual_feasibility=kkt.dual_feasibility,
        active_constraints=kkt.active_constraints,
        multipliers=kkt.multipliers,
        kkt_message=kkt.message,
    )


def _kkt_at(context: _SolveContext, guard: _GuardedProblem,
            u: np.ndarray) -> KKTReport:
    """KKT check in the optimizer's own scaled unit-box space."""

    if not context.policy_space.free_specs:
        return _unknown_kkt("no free design variable; nothing was optimized")
    _, g, ok = guard.primal(u)
    if not ok:
        return _unknown_kkt(
            "compiled primal is not finite at the returned point: "
            + guard.last_failure
        )
    grad_f, grad_ok = guard.gradient(u)
    jac, jac_ok = (
        guard.jacobian(u) if context.enforced_idx
        else (np.zeros((0, u.size)), True)
    )
    if not (grad_ok and jac_ok):
        return _unknown_kkt("derivatives unavailable at the returned point")
    return kkt_check(
        grad_f, jac, g, u,
        constraint_names=tuple(context.enforced),
        variable_names=tuple(spec.name for spec in context.policy_space.free_specs),
    )


# --------------------------------------------------------------------------- #
# Multi-start (R4)                                                            #
# --------------------------------------------------------------------------- #
def _start_points(reference: np.ndarray, n_starts: int,
                  seed: int) -> list[np.ndarray]:
    """Start 0 is the reference; the rest are scrambled Sobol unit points.

    Sobol points are drawn as a power-of-two block (their balance property)
    and truncated; the sequence is reproducible from ``seed``.
    """

    starts = [np.asarray(reference, dtype=float)]
    extra = n_starts - 1
    if extra > 0 and reference.size:
        m = int(np.ceil(np.log2(extra))) if extra > 1 else 0
        try:
            sobol = qmc.Sobol(d=reference.size, scramble=True, rng=int(seed))
        except TypeError:
            # SciPy < 1.15 names the generator argument ``seed``.
            sobol = qmc.Sobol(d=reference.size, scramble=True, seed=int(seed))
        starts.extend(np.asarray(p, dtype=float)
                      for p in sobol.random_base2(m)[:extra])
    return starts


def _passes_gates(result: NLPResult) -> bool:
    return (result.optimizer_status == "pass"
            and result.numerical_status == "pass")


def _select_start(results: list[NLPResult]) -> int:
    """Best passing start, else the least-violating one.

    Passing means optimizer and numerical status both pass.  Among passing
    starts a physics pass is preferred, then a converged optimizer run, then
    the lowest objective.  With no passing start, the smallest scaled
    violation wins (non-finite counts as infinite) and ties keep the earlier
    start, so the reference start is the default.
    """

    passing = [i for i, r in enumerate(results) if _passes_gates(r)]
    if passing:
        return min(passing, key=lambda i: (
            results[i].physics_status != "pass",
            not results[i].success,
            results[i].objective_mass,
            i,
        ))

    def violation(i: int) -> float:
        value = results[i].max_violation
        return value if np.isfinite(value) else float("inf")

    return min(range(len(results)), key=lambda i: (violation(i), i))


def _basin_labels(results: list[NLPResult],
                  finals: list[np.ndarray]) -> list[int | None]:
    """Single-linkage clusters of converged, gate-passing starts."""

    members = [i for i, r in enumerate(results)
               if r.success and _passes_gates(r)
               and np.isfinite(r.objective_mass)]
    parent = {i: i for i in members}

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a_pos, a in enumerate(members):
        for b in members[a_pos + 1:]:
            distance = float(np.max(np.abs(finals[a] - finals[b]), initial=0.0))
            fa, fb = results[a].objective_mass, results[b].objective_mass
            rel = abs(fa - fb) / max(abs(fa), abs(fb), 1e-300)
            if distance <= _BASIN_DISTANCE_TOL and rel <= _BASIN_OBJECTIVE_RTOL:
                parent[root(b)] = root(a)
    labels: list[int | None] = [None] * len(results)
    ids: dict[int, int] = {}
    for i in members:
        labels[i] = ids.setdefault(root(i), len(ids))
    return labels


def _start_summary(index: int, start: np.ndarray, final: np.ndarray,
                   result: NLPResult, basin: int | None) -> dict:
    return {
        "start": index,
        "u_start": [float(v) for v in start],
        "u_final": [float(v) for v in final],
        "success": bool(result.success),
        "optimizer_status": result.optimizer_status,
        "numerical_status": result.numerical_status,
        "physics_status": result.physics_status,
        "objective_mass": float(result.objective_mass),
        "max_violation": float(result.max_violation),
        "n_iter": int(result.n_iter),
        "kkt_status": result.kkt_status,
        "kkt_stationarity": float(result.kkt_stationarity),
        "evaluation_failures": int(result.evaluation_failures),
        "derivative_fallbacks": int(result.derivative_fallbacks),
        "basin": basin,
        "message": result.message,
    }


def _validate_count(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer, not {type(value).__name__}")
    if int(value) < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def solve_min_mass(mission: MissionSpec, isp_min: float, *,
                   u0: np.ndarray | None = None, couple_eta_cstar: bool = False,
                   enforced: tuple = DEFAULT_ENFORCED,
                   method: str = "SLSQP", maxiter: int = 150,
                   allow_incomplete_physics: bool = False,
                   objective: MassObjective | str = DEFAULT_MASS_OBJECTIVE,
                   contour_provider: ContourProviderSelection = None,
                   fixed_variables: Mapping[str, float] | None = None,
                   bounds: Mapping[str, tuple[float, float]] | None = None,
                   n_starts: int = 1,
                   start_seed: int = 0,
                   topology: GridTopology | None = None,
                   ) -> NLPResult:
    """Minimise mass subject to every manifest hard row.

    Authoritative runs fail preflight when an applicable coolant-side model is
    unavailable.  ``allow_incomplete_physics=True`` is an explicit screening
    mode; it may optimize the covered rows, but ``physics_status`` remains
    ``"unknown"`` and the compatibility ``feasible`` boolean remains false.

    ``fixed_variables`` substitutes physical values throughout the solve.
    ``bounds`` narrows the remaining variables' mission search windows. ``u0``
    contains only the remaining free coordinates in their reduced unit box;
    ``NLPResult.x`` still contains all physical mission coordinates for replay.
    An all-fixed request evaluates that design and its gates without an NLP.

    ``n_starts`` > 1 adds ``n_starts − 1`` scrambled Sobol starts (seeded by
    ``start_seed``) to the reference start ``u0``.  The compiled programs are
    built once and reused.  The returned result is the best start that passes
    optimizer and numerical status (physics pass preferred), else the
    least-violating one; ``start_summaries`` and ``n_basins`` record all.

    A numerical failure at a trial point is rejected inside the optimizer
    (see :class:`_GuardedProblem`); a failure at the returned point gives
    ``success=False`` and ``numerical_status="fail"``.  A returned point that
    the independent KKT check does not confirm, or an iterate where no
    derivative exists, also gives ``success=False`` / ``workflow_status
    "fail"``.  Configuration errors (``ValueError``, ``TypeError``,
    :class:`ModelCoverageError`) still raise.

    ``topology`` re-solves the whole problem on a non-default station grid
    (used by :func:`raosim.mdo.convergence.reoptimized_refinement_study`).
    """
    selected_objective = coerce_mass_objective(objective)
    if isinstance(isp_min, (bool, np.bool_)) or not np.isfinite(isp_min) or isp_min < 0:
        raise ValueError("isp_min must be a finite nonnegative number, not bool")
    n_starts = _validate_count("n_starts", n_starts, 1)
    start_seed = _validate_count("start_seed", start_seed, 0)
    policy_space = _OptimizationSpace.for_mission(mission, fixed_variables, bounds)
    # Loading here validates table schema, content identity, propellant pair and
    # meaningful O/F dependence before a path string can grant model coverage.
    surfaces = chamber_surfaces_for(mission)
    coolant = get_propellant(mission.propellant_name).coolant_name
    require_htd_coverage(
        coolant,
        has_real_fluid_properties=False,
        allow_incomplete_physics=allow_incomplete_physics,
    )
    space = policy_space.free_specs
    unknown = sorted(set(enforced) - set(CONSTRAINT_NAMES))
    if unknown:
        raise ValueError(f"unknown enforced constraint names: {unknown}")
    enforced_idx = tuple(CONSTRAINT_NAMES.index(n) for n in enforced)
    ss, obj, obj_grad, con, con_jac = _make_callables(
        mission,
        isp_min,
        couple_eta_cstar,
        enforced_idx,
        selected_objective,
        surfaces,
        contour_provider,
        optimization_space=policy_space,
        topology=topology,
    )
    if u0 is None:
        ref = jnp.asarray([s.ref() for s in space], dtype=jnp.float64)
        u0 = np.asarray(ss.to_unit(ref))
    u0 = np.asarray(u0, dtype=float)
    if u0.shape != (len(space),) or not np.isfinite(u0).all():
        raise ValueError(
            f"u0 must contain {len(space)} finite free-variable unit coordinates"
        )
    u0 = np.clip(u0, 0.0, 1.0)

    context = _SolveContext(
        mission=mission,
        isp_min=float(isp_min),
        enforced=tuple(enforced),
        enforced_idx=enforced_idx,
        policy_space=policy_space,
        scaled_space=ss,
        surfaces=surfaces,
        couple_eta_cstar=couple_eta_cstar,
        contour_provider=contour_provider,
        objective=selected_objective,
        obj=obj,
        obj_grad=obj_grad,
        con=con,
        con_jac=con_jac,
        method=method,
        maxiter=maxiter,
        topology=topology,
    )
    # An all-fixed request has a single point to evaluate, whatever n_starts.
    starts = _start_points(u0, n_starts if space else 1, start_seed)
    results: list[NLPResult] = []
    finals: list[np.ndarray] = []
    for start in starts:
        guard = context.guard()
        try:
            res = _optimize(context, guard, start)
        except _DerivativeUnavailable as exc:
            # Grade the iterate where the run stopped; it is never a success.
            res = OptimizeResult(
                x=np.clip(exc.u, 0.0, 1.0), success=False, nit=-1,
                message=(
                    f"stopped: {exc}; last failure: {guard.last_failure}"
                ),
            )
        finals.append(np.clip(np.asarray(res.x, dtype=float), 0.0, 1.0))
        results.append(_assemble_result(context, res, guard))

    chosen = _select_start(results)
    basins = _basin_labels(results, finals)
    summaries = tuple(
        _start_summary(i, starts[i], finals[i], results[i], basins[i])
        for i in range(len(results))
    )
    return replace(
        results[chosen],
        evaluation_failures=sum(r.evaluation_failures for r in results),
        derivative_fallbacks=sum(r.derivative_fallbacks for r in results),
        derivative_failures=sum(r.derivative_failures for r in results),
        n_starts=len(results),
        selected_start=chosen,
        n_basins=len({b for b in basins if b is not None}),
        start_summaries=summaries,
    )


def _failed_frontier_point(
    mission: MissionSpec,
    isp_min: float,
    objective: MassObjective | str,
    enforced: tuple,
    message: str,
) -> NLPResult:
    nan = float("nan")
    names = DesignVector.names()
    return _failure_result(
        isp_min=float(isp_min),
        x_phys=np.full(len(default_design_space(mission)), nan),
        design={name: nan for name in names},
        objective_name=coerce_mass_objective(objective).value,
        enforced=tuple(enforced),
        n_iter=0,
        message=message,
    )


def pareto_frontier(mission: MissionSpec, isp_grid, *,
                    couple_eta_cstar: bool = False,
                    enforced: tuple = DEFAULT_ENFORCED,
                    allow_incomplete_physics: bool = False,
                    objective: MassObjective | str = DEFAULT_MASS_OBJECTIVE,
                    contour_provider: ContourProviderSelection = None,
                    fixed_variables: Mapping[str, float] | None = None,
                    bounds: Mapping[str, tuple[float, float]] | None = None,
                    maxiter: int = 150,
                    n_starts: int = 1,
                    start_seed: int = 0) -> list[NLPResult]:
    """ε-constraint sweep: min mass at each I_sp floor, warm-started along the
    frontier.  Returns one result per floor, in the order given.

    Each floor is isolated: a solve that fails (``success=False``) or raises
    a runtime error is recorded as a failed point and the sweep continues.
    Only a successful, optimizer- and numerically-passing point seeds the
    next warm start, so one bad floor cannot poison the rest of the frontier.
    Configuration errors (``ValueError``, ``TypeError``, ``KeyError``,
    :class:`ModelCoverageError`) describe the whole sweep and still raise.
    """
    policy_space = _OptimizationSpace.for_mission(mission, fixed_variables, bounds)
    ss = ScaledSpace.from_specs(policy_space.free_specs)
    out: list[NLPResult] = []
    u0 = None
    for isp_min in isp_grid:
        try:
            r = solve_min_mass(mission, float(isp_min), u0=u0,
                               couple_eta_cstar=couple_eta_cstar,
                               enforced=enforced, maxiter=maxiter,
                               allow_incomplete_physics=allow_incomplete_physics,
                               objective=objective,
                               contour_provider=contour_provider,
                               fixed_variables=fixed_variables, bounds=bounds,
                               n_starts=n_starts, start_seed=start_seed)
        except ModelCoverageError:
            raise
        except (ValueError, TypeError, KeyError):
            raise
        except (RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
            r = _failed_frontier_point(
                mission, float(isp_min), objective, enforced,
                f"solve raised at this floor: {_describe_failure(exc)}",
            )
        out.append(r)
        if r.success and _passes_gates(r):
            u0 = np.asarray(ss.to_unit(policy_space.select_free(r.x)))
    return out
