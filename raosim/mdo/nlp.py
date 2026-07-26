"""
raosim.mdo.nlp — Phase 8/9: the ε-constraint hard-constrained NLP + Pareto sweep.

Objective/constraint policy (plan §8):

    min  m_package        s.t.  I_sp ≥ I_sp,min          (the ε-constraint)
                                every *enforced* discipline margin ≥ 0

sweeping ``I_sp,min`` traces the mass–performance frontier.  The engine is
always thrust-closed internally (the outer Newton), so thrust is NOT a free NLP
constraint — the design variables are the six of ``DesignVector`` and every
evaluation returns a converged, differentiable engine (``mdo.engine``).

Derivatives (plan §4.3, §12.2 row 8): a *low-dimensional, constraint-rich*
problem (n_x = 9, 8 constraints), so total derivatives are exact through the two
IFT solves and handed to SLSQP with no finite-difference step noise.  The
constraint Jacobian is assembled by **reverse mode** (``jax.jacrev``) and the
scalar objective by ``jax.grad``: forward mode (``jacfwd``) through the jit'd
nested Optimistix root-finds drops the tangent of the geometry-only min-
aggregated margins (a forward-through-implicit-solve quirk), and at 8×9 reverse
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
from raosim.mdo.engine import solve_engine

Array = jnp.ndarray

#: inequality order (all scaled so ≥0 = feasible)
CONSTRAINT_NAMES = (
    "isp_epsilon", "separation", "coking", "land_fit", "chug",
    "pintle_transition", "pump_suction", "pump_tip_speed", "aspect_ratio",
    "blockage_lo", "blockage_hi", "thermal_stress", "wall_temp",
)
#: enforced by default — ALL constraints, including coking, which is now
#: satisfiable via the film-cooling design variable (see module docstring).
DEFAULT_ENFORCED = CONSTRAINT_NAMES

# per-constraint reference scales (bring each margin to ~O(1) for the QP)
_C_SCALE = np.array([50.0, 1.0, 300.0, 5.0e-4, 0.2, 5.0e-5, 2.0, 300.0, 8.0,
                     0.3, 0.3, 1.0e8, 100.0])
_MASS_REF = 50.0  # kg, objective conditioning


def _constraint_vector(r, isp_min: float) -> Array:
    """Full scaled inequality vector g(x) ≥ 0 from a solved EngineResult."""
    c = r.constraints
    raw = jnp.stack([
        r.Isp - isp_min,
        c["separation_margin"],
        c["coking_margin_min"],
        c["land_min"],
        c["chug_margin_min"],
        c["pintle_transition_margin"],
        c["nss_margin_min"],
        c["tip_speed_margin_min"],
        c["aspect_ratio_margin"],
        c["blockage_lo_margin"],
        c["blockage_hi_margin"],
        c["thermal_stress_margin"],
        c["wall_temp_margin"],
    ])
    return raw / jnp.asarray(_C_SCALE)


def _raw_margins(r, isp_min: float) -> dict:
    c = r.constraints
    return {
        "isp_epsilon": float(r.Isp - isp_min),
        "separation": float(c["separation_margin"]),
        "coking": float(c["coking_margin_min"]),
        "land_fit": float(c["land_min"]),
        "chug": float(c["chug_margin_min"]),
        "pintle_transition": float(c["pintle_transition_margin"]),
        "pump_suction": float(c["nss_margin_min"]),
        "pump_tip_speed": float(c["tip_speed_margin_min"]),
        "aspect_ratio": float(c["aspect_ratio_margin"]),
        "blockage_lo": float(c["blockage_lo_margin"]),
        "blockage_hi": float(c["blockage_hi_margin"]),
        "thermal_stress": float(c["thermal_stress_margin"]),
        "wall_temp": float(c["wall_temp_margin"]),
    }


@dataclass(frozen=True)
class NLPResult:
    success: bool
    isp_min: float
    x: np.ndarray                 # physical 6-vector
    design: dict                  # named physical variables
    package_mass: float
    Isp: float
    constraints: dict             # named RAW margins (≥0 feasible), ALL reported
    enforced: tuple               # which were hard constraints
    max_violation: float          # max scaled violation over the ENFORCED set
    feasible: bool
    n_iter: int
    message: str


def _make_callables(mission, isp_min, couple, enforced_idx):
    space = default_design_space(mission)
    ss = ScaledSpace.from_specs(space)
    idx = jnp.asarray(enforced_idx, dtype=int)

    def solve(u):
        return solve_engine(DesignVector.from_array(ss.to_physical(u)),
                            mission, couple_eta_cstar=couple)

    @jax.jit
    def obj(u):
        return solve(u).package_mass / _MASS_REF

    @jax.jit
    def con(u):
        return _constraint_vector(solve(u), isp_min)[idx]

    # Reverse-mode (jacrev) for the constraint Jacobian: forward-mode (jacfwd)
    # through the jit'd nested Optimistix root-finds drops the tangent of the
    # geometry-only min-aggregated margins (e.g. land_min) — a forward-through-
    # implicit-solve quirk; jacrev is exact (FD-verified).  At 8 constraints × 9
    # variables reverse mode costs no more than forward here anyway.
    return ss, obj, jax.jit(jax.grad(obj)), con, jax.jit(jax.jacrev(con))


def solve_min_mass(mission: MissionSpec, isp_min: float, *,
                   u0: np.ndarray | None = None, couple_eta_cstar: bool = False,
                   enforced: tuple = DEFAULT_ENFORCED,
                   method: str = "SLSQP", maxiter: int = 150) -> NLPResult:
    """Minimise package mass at a fixed I_sp floor with exact Jacobians."""
    space = default_design_space(mission)
    enforced_idx = tuple(CONSTRAINT_NAMES.index(n) for n in enforced)
    ss, obj, obj_grad, con, con_jac = _make_callables(
        mission, isp_min, couple_eta_cstar, enforced_idx)
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
    r = solve_engine(DesignVector.from_array(jnp.asarray(x_phys)), mission,
                     couple_eta_cstar=couple_eta_cstar)
    scaled_all = np.asarray(_constraint_vector(r, isp_min))
    viol = float(max(0.0, -min(scaled_all[i] for i in enforced_idx)))
    names = [s.name for s in space]
    return NLPResult(
        success=bool(res.success), isp_min=float(isp_min), x=x_phys,
        design=dict(zip(names, (float(v) for v in x_phys))),
        package_mass=float(r.package_mass), Isp=float(r.Isp),
        constraints=_raw_margins(r, isp_min), enforced=tuple(enforced),
        max_violation=viol, feasible=viol < 1e-5,
        n_iter=int(getattr(res, "nit", -1)), message=str(res.message),
    )


def pareto_frontier(mission: MissionSpec, isp_grid, *,
                    couple_eta_cstar: bool = False,
                    enforced: tuple = DEFAULT_ENFORCED,
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
                           enforced=enforced, maxiter=maxiter)
        out.append(r)
        if r.success:
            u0 = np.asarray(ss.to_unit(jnp.asarray(r.x)))
    return out
