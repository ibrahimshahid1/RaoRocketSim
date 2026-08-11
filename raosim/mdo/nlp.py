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
constraint — the design variables are the ten of ``DesignVector`` and every
evaluation returns a converged, differentiable engine (``mdo.engine``).

Derivatives (plan §4.3, §12.2 row 8): a *low-dimensional, constraint-rich*
problem (n_x = 10, 17 constraints), so total derivatives are exact through the two
IFT solves and handed to SLSQP with no finite-difference step noise.  The
constraint Jacobian is assembled by **reverse mode** (``jax.jacrev``) and the
scalar objective by ``jax.grad``: forward mode (``jacfwd``) through the jit'd
nested Optimistix root-finds drops the tangent of the geometry-only min-
aggregated margins (a forward-through-implicit-solve quirk), and at 17×10 reverse
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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import raosim.jax  # noqa: F401  -- float64
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

from raosim.mdo.schema import DesignVector, MissionSpec, default_design_space
from raosim.mdo.scaling import ScaledSpace
from raosim.mdo.engine import chamber_surfaces_for, solve_engine
from raosim.mdo.constraints import (
    OPTIMIZER_CONSTRAINT_NAMES,
    OPTIMIZER_CONSTRAINT_SCALES,
    OPTIMIZER_CONSTRAINT_SPECS,
)
from raosim.mdo.coolant_htd import require_htd_coverage
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


def _constraint_vector(r, isp_min: float) -> Array:
    """Full scaled inequality vector g(x) ≥ 0 from a solved EngineResult."""
    raw = jnp.stack([
        (r.Isp - isp_min) if spec.engine_key is None
        else r.constraints[spec.engine_key]
        for spec in OPTIMIZER_CONSTRAINT_SPECS
    ])
    return raw / jnp.asarray(_C_SCALE)


def _raw_margins(r, isp_min: float) -> dict:
    return {
        spec.name: float(
            r.Isp - isp_min if spec.engine_key is None
            else r.constraints[spec.engine_key]
        )
        for spec in OPTIMIZER_CONSTRAINT_SPECS
    }


@dataclass(frozen=True)
class NLPResult:
    success: bool
    isp_min: float
    x: np.ndarray                 # physical active optimizer vector (10/11)
    design: dict                  # fixed 11-value physical contract
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

    @property
    def package_mass(self) -> float:
        """Deprecated alias for the smooth optimization objective."""

        return self.objective_mass


def _make_callables(
    mission,
    isp_min,
    couple,
    enforced_idx,
    objective=DEFAULT_MASS_OBJECTIVE,
    surfaces=None,
):
    objective = coerce_mass_objective(objective)
    if surfaces is None:
        # Backward-compatible internal/testing entry point.  Resolution still
        # performs the same schema, propellant-identity, and domain validation
        # as the public solve.
        surfaces = chamber_surfaces_for(mission)
    space = default_design_space(mission)
    layout = mission.design_layout()
    ss = ScaledSpace.from_specs(space)
    idx = jnp.asarray(enforced_idx, dtype=int)

    def solve(u):
        return solve_engine(
            DesignVector.from_active_array(
                ss.to_physical(u), layout, fixed_of=mission.OF
            ),
            mission,
            couple_eta_cstar=couple,
            surfaces=surfaces,
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


def solve_min_mass(mission: MissionSpec, isp_min: float, *,
                   u0: np.ndarray | None = None, couple_eta_cstar: bool = False,
                   enforced: tuple = DEFAULT_ENFORCED,
                   method: str = "SLSQP", maxiter: int = 150,
                   allow_incomplete_physics: bool = False,
                   objective: MassObjective | str = DEFAULT_MASS_OBJECTIVE,
                   ) -> NLPResult:
    """Minimise mass subject to every manifest hard row.

    Authoritative runs fail preflight when an applicable coolant-side model is
    unavailable.  ``allow_incomplete_physics=True`` is an explicit screening
    mode; it may optimize the covered rows, but ``physics_status`` remains
    ``"unknown"`` and the compatibility ``feasible`` boolean remains false.
    """
    selected_objective = coerce_mass_objective(objective)
    # Loading here validates table schema, content identity, propellant pair and
    # meaningful O/F dependence before a path string can grant model coverage.
    surfaces = chamber_surfaces_for(mission)
    coolant = get_propellant(mission.propellant_name).coolant_name
    htd_available, _ = require_htd_coverage(
        coolant,
        has_real_fluid_properties=False,
        allow_incomplete_physics=allow_incomplete_physics,
    )
    space = default_design_space(mission)
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
    )
    if u0 is None:
        ref = jnp.asarray([s.ref() for s in space], dtype=jnp.float64)
        u0 = np.asarray(ss.to_unit(ref))
    u0 = np.clip(np.asarray(u0, dtype=float), 0.0, 1.0)

    f = lambda u: float(obj(jnp.asarray(u)))
    fg = lambda u: np.asarray(obj_grad(jnp.asarray(u)), dtype=float)
    cf = lambda u: np.asarray(con(jnp.asarray(u)), dtype=float)
    cj = lambda u: np.asarray(con_jac(jnp.asarray(u)), dtype=float)

    res = minimize(f, u0, jac=fg, method=method,
                   bounds=[(0.0, 1.0)] * len(space),
                   constraints=[{"type": "ineq", "fun": cf, "jac": cj}],
                   options={"maxiter": maxiter, "ftol": 1e-9})

    u = np.clip(res.x, 0.0, 1.0)
    x_phys = np.asarray(ss.to_physical(jnp.asarray(u)))
    layout = mission.design_layout()
    design_vector = DesignVector.from_active_array(
        jnp.asarray(x_phys), layout, fixed_of=mission.OF
    )
    r = solve_engine(
        design_vector,
        mission,
        couple_eta_cstar=couple_eta_cstar,
        surfaces=surfaces,
    )
    scaled_all = np.asarray(_constraint_vector(r, isp_min))
    viol = float(max(0.0, -min(scaled_all[i] for i in enforced_idx)))
    optimizer_status = "pass" if viol < 1e-5 else "fail"
    numerical_status = (
        "pass" if bool(r.solver_converged) and bool(r.finite) else "fail"
    )
    requirement_indices = tuple(
        i
        for i, spec in enumerate(OPTIMIZER_CONSTRAINT_SPECS)
        if spec.category == "requirement"
    )
    physical_indices = tuple(
        i
        for i, spec in enumerate(OPTIMIZER_CONSTRAINT_SPECS)
        if spec.category not in {"requirement", "numerical"}
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
    physical_row_failed = any(
        np.isfinite(scaled_all[i]) and scaled_all[i] < -1.0e-5
        for i in physical_indices
    )
    if physical_row_failed:
        physics_status = "fail"
    elif numerical_status != "pass" or any(
        not np.isfinite(scaled_all[i]) for i in physical_indices
    ):
        # A failed numerical evaluation cannot establish a physical pass/fail.
        physics_status = "unknown"
    else:
        physics_status = "pass" if htd_available else "unknown"
    workflow_ok = bool(res.success) and numerical_status == "pass"
    workflow_status = "pass" if workflow_ok else "fail"
    optimizer_feasible = optimizer_status == "pass"
    complete_pass = (
        workflow_status == "pass"
        and optimizer_status == "pass"
        and numerical_status == "pass"
        and physics_status == "pass"
        and requirements_status == "pass"
    )
    return NLPResult(
        success=workflow_ok,
        isp_min=float(isp_min),
        x=x_phys,
        design={
            name: float(value)
            for name, value in design_vector.as_contract_dict(
                effective_of=r.OF
            ).items()
        },
        objective_name=selected_objective.value,
        objective_mass=float(mass_objective_value(r, selected_objective)),
        electric_package_objective_mass=float(
            r.electric_package_objective_mass
        ),
        dry_mass_partial_objective_mass=float(
            r.dry_mass_partial_objective_mass
        ),
        exact_electric_package_mass=float(r.electric_package_exact_mass),
        exact_dry_mass_partial=float(r.dry_mass_partial_exact_mass),
        Isp=float(r.Isp),
        constraints=_raw_margins(r, isp_min), enforced=tuple(enforced),
        max_violation=viol,
        optimizer_status=optimizer_status,
        numerical_status=numerical_status,
        requirements_status=requirements_status,
        optimizer_feasible=optimizer_feasible,
        physics_status=physics_status,
        workflow_status=workflow_status,
        feasible=complete_pass,
        n_iter=int(getattr(res, "nit", -1)), message=str(res.message),
    )


def pareto_frontier(mission: MissionSpec, isp_grid, *,
                    couple_eta_cstar: bool = False,
                    enforced: tuple = DEFAULT_ENFORCED,
                    allow_incomplete_physics: bool = False,
                    objective: MassObjective | str = DEFAULT_MASS_OBJECTIVE,
                    maxiter: int = 150) -> list[NLPResult]:
    """ε-constraint sweep: min mass at each I_sp floor, warm-started along the
    frontier.  Returns results in ascending I_sp order."""
    space = default_design_space(mission)
    ss = ScaledSpace.from_specs(space)
    out: list[NLPResult] = []
    u0 = None
    for isp_min in isp_grid:
        r = solve_min_mass(mission, float(isp_min), u0=u0,
                           couple_eta_cstar=couple_eta_cstar,
                           enforced=enforced, maxiter=maxiter,
                           allow_incomplete_physics=allow_incomplete_physics,
                           objective=objective)
        out.append(r)
        if r.success:
            u0 = np.asarray(ss.to_unit(jnp.asarray(r.x)))
    return out
