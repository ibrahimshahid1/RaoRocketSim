"""
raosim.jax.design_opt — constrained differentiable nozzle design.

Maximize the ambient thrust coefficient over a small contour design
vector subject to the physics constraints, with EXACT JAX gradients
(Optimistix BFGS on a quadratic-penalty objective):

    maximize   Cf(ε; Pc, Pa)
    over       ε (expansion ratio), Rd_factor (throat curvature)
    s.t.       separation_margin(ε) ≥ sep_margin_min        (Schmucker)
               T_wg,throat(Rd_factor) ≤ T_wall_limit        (Bartz + Sieder-Tate)

This is the rigorous form of the coupling the screening workflow
lacks: the physics SHAPES the geometry (ε set by the thrust/separation
trade, Rd_factor by cooling) instead of being checked after a fixed
Bézier.  It is the differentiable successor to the NumPy feedback loop
in :mod:`raosim.thermal_design`.

The gas properties (cp, Pr, μ) and the coolant-side h_c are solve
constants (independent of ε / Rd_factor) precomputed on the host; only
the inner objective is differentiated.  Design variables are kept in
their boxes by a sigmoid reparametrisation, and the penalty weight is
ramped for accuracy — the reported margins are the unweighted physics.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

import raosim.jax  # noqa: F401  -- enables x64
import jax
import jax.numpy as jnp
import optimistix as optx

from raosim.jax import thermal as T


def _sigmoid_box(z, lo, hi):
    return lo + (hi - lo) * jax.nn.sigmoid(z)


def constrained_nozzle_design(
    *,
    Rt: float,
    Pc: float,
    Pa: float,
    gamma: float,
    c_star: float,
    Tc: float,
    cp_gas: float,
    Pr_gas: float,
    mu_gas: float,
    h_c: float,
    coolant_temperature: float,
    t_wall: float,
    k_wall: float,
    T_wall_limit: float,
    q_limit: float | None = None,
    sep_margin_min: float = 1.2,  # SP-8120 rule: Pe ≥ 1.2·p_sep ("within 20 %")
    eps_bounds: tuple[float, float] = (2.0, 50.0),
    rd_factor_bounds: tuple[float, float] = (0.382, 5.0),
    eps0: float | None = None,
    rd_factor0: float | None = None,
    penalty_weights: tuple[float, ...] = (1e1, 1e3, 1e5),
    max_steps: int = 200,
) -> dict:
    """Solve the constrained design.  Returns the optimal ``epsilon`` /
    ``Rd_factor``, the achieved ``Cf``, ``separation_margin``,
    ``throat_wall_temperature``, ``throat_heat_flux``, which constraints
    are active, and convergence info.
    """
    eps_lo, eps_hi = eps_bounds
    rd_lo, rd_hi = rd_factor_bounds
    eps0 = float(eps0 if eps0 is not None else 0.5 * (eps_lo + eps_hi))
    rd0 = float(rd_factor0 if rd_factor0 is not None else rd_lo)

    def _logit(p):
        p = min(max(p, 1e-3), 1.0 - 1e-3)
        return float(np.log(p) - np.log1p(-p))

    z0 = jnp.array([
        _logit((eps0 - eps_lo) / (eps_hi - eps_lo)),
        _logit((rd0 - rd_lo) / (rd_hi - rd_lo)),
    ])

    def physics(z):
        eps = _sigmoid_box(z[0], eps_lo, eps_hi)
        rd_factor = _sigmoid_box(z[1], rd_lo, rd_hi)
        Cf = T.ambient_thrust_coefficient(eps, gamma, Pc, Pa)
        sep = T.schmucker_separation_margin(eps, gamma, Pc, Pa)
        T_wg, q = T.throat_wall_temperature(
            Rt=Rt, Pc=Pc, c_star=c_star, cp_gas=cp_gas, Pr_gas=Pr_gas,
            mu_gas=mu_gas, gamma=gamma, Tc=Tc,
            throat_curvature_radius=rd_factor * Rt,
            coolant_temperature=coolant_temperature, h_c=h_c,
            t_wall=t_wall, k_wall=k_wall,
        )
        return eps, rd_factor, Cf, sep, T_wg, q

    q_lim = float(q_limit) if q_limit is not None else None

    def objective(z, w):
        eps, rd_factor, Cf, sep, T_wg, q = physics(z)
        # Normalised constraint violations (>0 = violated).
        g_sep = jnp.maximum(sep_margin_min - sep, 0.0) / max(sep_margin_min, 1e-9)
        g_cool = jnp.maximum(T_wg - T_wall_limit, 0.0) / T_wall_limit
        pen = g_sep ** 2 + g_cool ** 2
        if q_lim is not None:
            g_q = jnp.maximum(q - q_lim, 0.0) / q_lim
            pen = pen + g_q ** 2
        return -Cf + w * pen

    solver = optx.BFGS(rtol=1e-8, atol=1e-10)
    z = z0
    n_steps = 0
    success = False
    for w in penalty_weights:
        sol = optx.minimise(
            objective, solver, z, args=jnp.asarray(float(w)),
            max_steps=max_steps, throw=False,
        )
        z = sol.value
        success = bool(sol.result == optx.RESULTS.successful)
        n_steps += int(sol.stats.get("num_steps", 0))

    eps, rd_factor, Cf, sep, T_wg, q = (float(v) for v in physics(z))
    sep_active = sep <= sep_margin_min * (1.0 + 1e-3)
    cool_active = T_wg >= T_wall_limit * (1.0 - 1e-3)
    q_active = q_lim is not None and q >= q_lim * (1.0 - 1e-3)
    feasible = (sep >= sep_margin_min * (1.0 - 1e-3)
                and T_wg <= T_wall_limit * (1.0 + 1e-3)
                and (q_lim is None or q <= q_lim * (1.0 + 1e-3)))
    return SimpleNamespace(
        epsilon=eps,
        Rd_factor=rd_factor,
        Cf=Cf,
        separation_margin=sep,
        throat_wall_temperature=T_wg,
        throat_heat_flux=q,
        separation_active=bool(sep_active),
        cooling_active=bool(cool_active),
        heat_flux_active=bool(q_active),
        feasible=bool(feasible),
        success=success,
        n_steps=n_steps,
        backend="jax",
    )


def design_gradients(design_args: dict, eps: float, rd_factor: float) -> dict:
    """Exact ∂(Cf, sep_margin, T_wg)/∂(ε, Rd_factor) at a design point —
    the sensitivity field the screening workflow cannot produce.
    ``design_args`` is the kwargs dict passed to
    :func:`constrained_nozzle_design` (only the physics constants are
    read)."""
    a = design_args

    def outs(v):
        e, rd = v
        Cf = T.ambient_thrust_coefficient(e, a["gamma"], a["Pc"], a["Pa"])
        sep = T.schmucker_separation_margin(e, a["gamma"], a["Pc"], a["Pa"])
        T_wg, _ = T.throat_wall_temperature(
            Rt=a["Rt"], Pc=a["Pc"], c_star=a["c_star"], cp_gas=a["cp_gas"],
            Pr_gas=a["Pr_gas"], mu_gas=a["mu_gas"], gamma=a["gamma"],
            Tc=a["Tc"], throat_curvature_radius=rd * a["Rt"],
            coolant_temperature=a["coolant_temperature"], h_c=a["h_c"],
            t_wall=a["t_wall"], k_wall=a["k_wall"],
        )
        return jnp.stack([Cf, sep, T_wg])

    J = jax.jacfwd(outs)(jnp.array([float(eps), float(rd_factor)]))
    J = np.asarray(J)
    names = ["Cf", "separation_margin", "throat_wall_temperature"]
    return {n: {"d_d_epsilon": float(J[i, 0]),
                "d_d_Rd_factor": float(J[i, 1])}
            for i, n in enumerate(names)}


# --------------------------------------------------------------------------- #
# Thrust-targeted, multi-objective constrained design.                         #
#                                                                             #
# Extension of constrained_nozzle_design: instead of a fixed throat radius     #
# and a single Cf objective, the THRUST is a hard spec that sizes the throat,  #
# and the user selects one or more performance/size objectives that the        #
# (still exact-gradient) optimiser trades off.  Every relation below is         #
# grounded in propulsion_texts:                                                 #
#                                                                             #
#   F = C_F · P_c · A_t   ->   A_t = F/(C_F·P_c),  R_t = sqrt(A_t/pi)           #
#        Huzel & Huang, NASA SP-125, Eq. 1-31/1-33  (19710019929.pdf)          #
#   Isp = C_F · c* / g0                              SP-125 Eq. 1-31c           #
#   L_n = (L%/100)·(R_e − R_t)/tan(15°)             Rao 1958/61; NASA SP-8120   #
#        (R_e = sqrt(eps)·R_t)                       (nozzle_geometry.py)       #
#   Constraints: Schmucker separation (Östlund, fulltext01.pdf §6.3.1),        #
#   Bartz gas-side + Sieder-Tate coolant-side wall temperature (SP-125 §4).    #
# --------------------------------------------------------------------------- #
G0 = 9.80665  # m/s^2  standard gravity (Isp = Cf·c*/g0; SP-125 Eq. 1-31c)
_TAN15 = math.tan(math.radians(15.0))  # 15° reference-cone (Rao/SP-8120 L_n)

#: Objectives that are MAXIMISED (performance) vs MINIMISED (size).
_MAXIMISE_OBJECTIVES = ("isp", "cf")
_MINIMISE_OBJECTIVES = ("length", "mass")
OBJECTIVE_KEYS = _MAXIMISE_OBJECTIVES + _MINIMISE_OBJECTIVES


def normalize_objective_weights(objectives: dict | None) -> dict:
    """Validate/clean a ``{name: weight}`` objective map.

    Drops zero weights, rejects unknown names and negative weights, and
    lower-cases the keys.  An empty/None map means *feasibility-only*
    (no objective term; the penalty alone drives to a feasible design).
    """
    if not objectives:
        return {}
    out: dict[str, float] = {}
    for name, w in objectives.items():
        key = str(name).strip().lower()
        if key not in OBJECTIVE_KEYS:
            raise ValueError(
                f"unknown objective {name!r}; choose from {list(OBJECTIVE_KEYS)}")
        w = float(w)
        if w < 0.0:
            raise ValueError("objective weights must be >= 0")
        if w > 0.0:
            out[key] = out.get(key, 0.0) + w
    return out


def thrust_targeted_design(
    *,
    target_thrust: float,
    Pc: float,
    Pa: float,
    gamma: float,
    c_star: float,
    Tc: float,
    cp_gas: float,
    Pr_gas: float,
    mu_gas: float,
    h_c: float,
    coolant_temperature: float,
    t_wall: float,
    k_wall: float,
    T_wall_limit: float,
    length_pct: float = 80.0,
    wall_density: float = 8190.0,
    q_limit: float | None = None,
    sep_margin_min: float = 1.2,  # SP-8120 rule: Pe ≥ 1.2·p_sep ("within 20 %")
    objectives: dict | None = None,
    eps_bounds: tuple[float, float] = (2.0, 50.0),
    rd_factor_bounds: tuple[float, float] = (0.382, 5.0),
    eps0: float | None = None,
    rd_factor0: float | None = None,
    penalty_weights: tuple[float, ...] = (1e1, 1e3, 1e5),
    max_steps: int = 200,
) -> SimpleNamespace:
    """Size a nozzle for a FIXED THRUST TARGET while optimising a chosen
    (possibly multi-) objective, with exact JAX gradients.

    The thrust target is met *by construction*, not as a soft constraint:
    at every design point the throat area is set from the master thrust
    relation (SP-125 Eq. 1-31/1-33) ``A_t = F/(C_F·P_c)``, so ``R_t`` is a
    *dependent* variable carrying the spec.  The free differentiable
    design vector is ``(epsilon, Rd_factor)``; chamber pressure ``Pc`` and
    mixture ratio enter through the host-resolved ``gamma``/``c_star``/
    ``Tc`` (CEA is non-differentiable and stays on the host — see
    JAX_DIFFERENTIABLE_PLAN.md §3 — so optimising over ``Pc``/MR is a host
    loop *around* this call).

    Selectable objectives (``objectives = {name: weight}``):

    * ``"isp"``    — maximise specific impulse ``Isp = C_F·c*/g0``
    * ``"cf"``     — maximise thrust coefficient ``C_F``
    * ``"length"`` — minimise bell length ``L_n``
    * ``"mass"``   — minimise a wall-material proxy (frustum lateral area
      × ``t_wall`` × ``wall_density``)

    Multiple objectives are combined by weighted scalarisation of values
    normalised at the seed design, so weights are dimensionless trade-offs
    (e.g. ``{"isp": 1, "length": 1}`` weights performance and compactness
    equally).  ``isp`` and ``cf`` coincide here because ``c*`` is fixed by
    the host thermochemistry; they differ once a ``Pc``/MR loop varies it.

    Constraints (identical to :func:`constrained_nozzle_design`): Schmucker
    separation margin ≥ ``sep_margin_min``; throat wall temperature ≤
    ``T_wall_limit`` (Bartz + Sieder-Tate); optional heat-flux cap
    ``q_limit``.  Returns the optimal design, the achieved objective
    breakdown, which constraints are active, and feasibility.
    """
    objs = normalize_objective_weights(objectives)
    eps_lo, eps_hi = eps_bounds
    rd_lo, rd_hi = rd_factor_bounds
    eps0 = float(eps0 if eps0 is not None else 0.5 * (eps_lo + eps_hi))
    rd0 = float(rd_factor0 if rd_factor0 is not None else rd_lo)
    F = float(target_thrust)
    if F <= 0.0:
        raise ValueError("target_thrust must be positive")
    if Pc <= 0.0:
        raise ValueError("Pc must be positive")
    if Pa < 0.0:
        raise ValueError("Pa must be non-negative")
    if gamma <= 1.0:
        raise ValueError("gamma must be greater than 1")
    if c_star <= 0.0 or Tc <= 0.0:
        raise ValueError("c_star and Tc must be positive")
    if h_c <= 0.0 or t_wall <= 0.0 or k_wall <= 0.0:
        raise ValueError("h_c, t_wall, and k_wall must be positive")
    if T_wall_limit <= 0.0 or wall_density <= 0.0 or length_pct <= 0.0:
        raise ValueError("T_wall_limit, wall_density, and length_pct must be positive")
    if not (1.0 < eps_lo < eps_hi):
        raise ValueError("eps_bounds must satisfy 1 < lower < upper")
    if not (0.0 < rd_lo < rd_hi):
        raise ValueError("rd_factor_bounds must satisfy 0 < lower < upper")
    if not (eps_lo <= eps0 <= eps_hi):
        raise ValueError("eps0 must lie inside eps_bounds")
    if not (rd_lo <= rd0 <= rd_hi):
        raise ValueError("rd_factor0 must lie inside rd_factor_bounds")
    if not penalty_weights or any(float(w) <= 0.0 for w in penalty_weights):
        raise ValueError("penalty_weights must contain positive values")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")

    def _logit(p):
        p = min(max(p, 1e-3), 1.0 - 1e-3)
        return float(np.log(p) - np.log1p(-p))

    z0 = jnp.array([
        _logit((eps0 - eps_lo) / (eps_hi - eps_lo)),
        _logit((rd0 - rd_lo) / (rd_hi - rd_lo)),
    ])

    def physics(z):
        eps = _sigmoid_box(z[0], eps_lo, eps_hi)
        rd_factor = _sigmoid_box(z[1], rd_lo, rd_hi)
        Cf = T.ambient_thrust_coefficient(eps, gamma, Pc, Pa)
        # Keep the differentiated calculation finite if an extreme ambient
        # pressure makes a trial design's ambient Cf non-positive.  Such a
        # point is explicitly constrained out below; it is never reported
        # as a valid thrust-sized nozzle.
        Cf_for_sizing = jnp.maximum(Cf, 1e-6)
        At = F / (Cf_for_sizing * Pc)             # SP-125 Eq. 1-31/1-33
        Rt = jnp.sqrt(At / jnp.pi)
        Isp = Cf * c_star / G0                    # SP-125 Eq. 1-31c
        Re = jnp.sqrt(eps) * Rt
        Ln = (length_pct / 100.0) * (Re - Rt) / _TAN15   # Rao/SP-8120
        slant = jnp.sqrt(Ln * Ln + (Re - Rt) ** 2)
        # Wall-material proxy: frustum lateral area × wall thickness × ρ.
        mass = jnp.pi * (Re + Rt) * slant * t_wall * wall_density
        sep = T.schmucker_separation_margin(eps, gamma, Pc, Pa)
        T_wg, q = T.throat_wall_temperature(
            Rt=Rt, Pc=Pc, c_star=c_star, cp_gas=cp_gas, Pr_gas=Pr_gas,
            mu_gas=mu_gas, gamma=gamma, Tc=Tc,
            throat_curvature_radius=rd_factor * Rt,
            coolant_temperature=coolant_temperature, h_c=h_c,
            t_wall=t_wall, k_wall=k_wall,
        )
        return {"eps": eps, "rd_factor": rd_factor, "Cf": Cf, "At": At,
                "Rt": Rt, "Isp": Isp, "Re": Re, "Ln": Ln, "mass": mass,
                "sep": sep, "T_wg": T_wg, "q": q}

    # Seed reference values normalise the heterogeneous-unit objectives so
    # the weights are dimensionless trade-offs.
    ref = {k: float(v) for k, v in physics(z0).items()}
    refs = {
        "cf": max(ref["Cf"], 1e-9),
        "isp": max(ref["Isp"], 1e-9),
        "length": max(ref["Ln"], 1e-12),
        "mass": max(ref["mass"], 1e-12),
    }

    q_lim = float(q_limit) if q_limit is not None else None

    def objective_value(p):
        J = 0.0
        if "cf" in objs:
            J = J - objs["cf"] * p["Cf"] / refs["cf"]
        if "isp" in objs:
            J = J - objs["isp"] * p["Isp"] / refs["isp"]
        if "length" in objs:
            J = J + objs["length"] * p["Ln"] / refs["length"]
        if "mass" in objs:
            J = J + objs["mass"] * p["mass"] / refs["mass"]
        return J

    def objective(z, w):
        p = physics(z)
        # Normalised constraint violations (>0 = violated).
        g_sep = jnp.maximum(sep_margin_min - p["sep"], 0.0) / max(sep_margin_min, 1e-9)
        g_cool = jnp.maximum(p["T_wg"] - T_wall_limit, 0.0) / T_wall_limit
        g_positive_cf = jnp.maximum(1e-6 - p["Cf"], 0.0) / 1e-6
        pen = g_sep ** 2 + g_cool ** 2 + g_positive_cf ** 2
        if q_lim is not None:
            g_q = jnp.maximum(p["q"] - q_lim, 0.0) / q_lim
            pen = pen + g_q ** 2
        return objective_value(p) + w * pen

    solver = optx.BFGS(rtol=1e-8, atol=1e-10)
    z = z0
    n_steps = 0
    success = False
    for w in penalty_weights:
        sol = optx.minimise(
            objective, solver, z, args=jnp.asarray(float(w)),
            max_steps=max_steps, throw=False,
        )
        z = sol.value
        success = bool(sol.result == optx.RESULTS.successful)
        n_steps += int(sol.stats.get("num_steps", 0))

    p = {k: float(v) for k, v in physics(z).items()}
    thrust = p["Cf"] * Pc * p["At"]              # == target by construction
    sep_active = p["sep"] <= sep_margin_min * (1.0 + 1e-3)
    cool_active = p["T_wg"] >= T_wall_limit * (1.0 - 1e-3)
    q_active = q_lim is not None and p["q"] >= q_lim * (1.0 - 1e-3)
    positive_cf = p["Cf"] > 1e-6
    feasible = (positive_cf
                and p["sep"] >= sep_margin_min * (1.0 - 1e-3)
                and p["T_wg"] <= T_wall_limit * (1.0 + 1e-3)
                and (q_lim is None or p["q"] <= q_lim * (1.0 + 1e-3)))
    objective_values = {
        "cf": p["Cf"],
        "isp": p["Isp"],
        "length": p["Ln"],
        "mass": p["mass"],
    }
    objective_terms = {}
    for name, weight in objs.items():
        sign = -1.0 if name in _MAXIMISE_OBJECTIVES else 1.0
        objective_terms[name] = sign * weight * objective_values[name] / refs[name]
    return SimpleNamespace(
        epsilon=p["eps"],
        Rd_factor=p["rd_factor"],
        Rt=p["Rt"],
        throat_radius=p["Rt"],
        At=p["At"],
        exit_radius=p["Re"],
        Cf=p["Cf"],
        Isp=p["Isp"],
        thrust=thrust,
        target_thrust=F,
        nozzle_length=p["Ln"],
        wall_mass_proxy=p["mass"],
        separation_margin=p["sep"],
        throat_wall_temperature=p["T_wg"],
        throat_heat_flux=p["q"],
        separation_active=bool(sep_active),
        cooling_active=bool(cool_active),
        heat_flux_active=bool(q_active),
        positive_thrust_coefficient=bool(positive_cf),
        feasible=bool(feasible),
        objectives=dict(objs),
        objective_values=objective_values,
        objective_terms=objective_terms,
        scalar_objective=float(sum(objective_terms.values())),
        success=success,
        n_steps=n_steps,
        backend="jax",
    )
