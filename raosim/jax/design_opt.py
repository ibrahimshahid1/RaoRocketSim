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
    sep_margin_min: float = 1.0,
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
