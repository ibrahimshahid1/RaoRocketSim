"""
raosim.jax.assembly — the full assembled Rao BVP residual in JAX (J2/J3b).

This is the JAX re-expression of the NumPy residual assembly:

    _unpack_bvp -> _rao_bvp_residual_groups -> RaoResidualGroups.flat()

i.e. exactly what ``scipy.optimize.least_squares`` minimises in
``rao_variational.solve_rao_bvp`` (rao_variational.py:3559).  The J2 gate
(JAX_DIFFERENTIABLE_PLAN.md §7) is 1e-8 parity of :func:`make_residual`'s
output against ``_scaled_rao_bvp_residual`` on identical packed unknown
vectors — proving the port changes numerics, not physics.

Faithfulness notes (each mirrors a specific NumPy code path):

* CE flow nodes clamp ``M`` at **1.001** (``_control_surface_flow_nodes``);
  kernel-BD / wall-interp nodes clamp at **1.000001**
  (``nasa_moc.bd_segment_at_fraction``, ``_interp_wall_at_fraction``).
  The two clamps are deliberately NOT unified here.
* ``x_ce`` is reconstructed from the left-Mach ODE ``dx = dr/tan(θ+μ)``
  with the NumPy fallback step ``1e-9`` when ``|tan| < 1e-12``
  (``_integrate_ce_x_from_left_mach``; M floor **1.000001** there).
* The kernel BD is a *static* polyline during the solve (it is built once
  by the NumPy seed path); only ``kernel_d_fraction`` is an unknown.  The
  B→D mass flux is therefore evaluated as a masked partial-segment sum,
  smooth in ``kernel_d_fraction`` — this replaces
  ``calc_mdot_bd``/``bd_segment_at_fraction`` without changing values.
* Mass-flux midpoint clamps: CE-side **1.001**
  (``rao_variational.curve_mass_flux``), BD-side **1.000001**
  (``nasa_moc.curve_mass_flux``); both floor midpoint r at 1e-12 and skip
  segments with ds <= 1e-12.
* Group order and scaling are byte-for-byte ``RaoResidualGroups.flat()``.

Static configuration (node counts, active blocks, targets, chart angles,
the kernel BD arrays) is baked into a closure by :func:`make_residual`;
the returned function is pure ``fn(u) -> r`` and jit/jacfwd/jacrev-safe.

Unsupported (raise at build time, not silently wrong): residual blocks
``stationarity``/``transversality`` (finite-difference reference blocks,
not in the default stack) and ``angle_boundary_mode != "free"``.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

import raosim.jax  # noqa: F401  -- enables x64
import jax.numpy as jnp

from raosim.jax.primitives import (
    prandtl_meyer,
    mstar_from_M,
    isentropic_density_ratio,
    isentropic_temperature_ratio,
)

_ONE_DEG = math.radians(1.0)
_CE_M_FLOOR = 1.001          # _control_surface_flow_nodes / curve_mass_flux
_NASA_M_FLOOR = 1.000001     # nasa_moc nodes, wall interp, left-Mach integrator

# Residual blocks the JAX assembly evaluates (the default stack).
SUPPORTED_BLOCKS = frozenset({
    "mass", "length", "moc_cplus", "moc_cminus", "ce_geometry",
    "regularization", "penalties", "algebraic_stationarity",
    "wall_endpoint", "wall_tangency", "cplus_ce_to_wall",
    "wall_intersection", "left_mach",
})


class StaticParams(NamedTuple):
    """Solve-constant data captured by :func:`make_residual`'s closure."""

    n_ce: int
    n_wall: int                 # 0 => couple_wall=False (legacy layout)
    gamma: float
    physics_weight: float
    L_target: float
    Re: float
    Rt: float
    Nx: float                   # wall point N (chart theta_N, static)
    Ny: float
    mdot_target_throat: float   # quasi-1D throat target (_target_mdot)
    bd_full_flux: float         # rao_variational.curve_mass_flux(kernel_bd)
    bd_x: np.ndarray            # kernel BD polyline, wall-first (B -> axis)
    bd_r: np.ndarray
    bd_M: np.ndarray
    bd_theta: np.ndarray
    active: frozenset


def params_from_config(config, physics_weight: float | None = None) -> StaticParams:
    """Build :class:`StaticParams` from a ``RaoSolverConfig``.

    Host-side (NumPy) — call once per solve.  ``config.kernel_bd`` must be
    populated (the public ``solve_rao_bvp`` path always populates it).
    ``physics_weight`` defaults to the *current* module value of
    ``rao_variational.PHYSICS_WEIGHT`` so monkeypatched studies (the
    weight-ramp tests) see the same weight on both backends.
    """
    import raosim.rao_variational as rv

    if not config.kernel_bd:
        raise ValueError(
            "JAX assembled residual requires config.kernel_bd (the static "
            "kernel BD polyline from the seed builder)."
        )
    mode = getattr(config, "angle_boundary_mode", "free")
    if mode != "free":
        raise NotImplementedError(
            f"angle_boundary_mode={mode!r} is not supported by the JAX "
            "backend (only 'free'; chart anchors would contaminate the "
            "Phase 7 benchmark anyway — REWRITE_PLAN.md §13.1)."
        )
    active = rv._enabled_residual_blocks(config)
    unsupported = active.difference(SUPPORTED_BLOCKS)
    if unsupported:
        raise NotImplementedError(
            f"Residual blocks {sorted(unsupported)} are not in the JAX "
            "assembly (finite-difference reference blocks stay NumPy-only)."
        )

    Rt = config.Rt
    Re = math.sqrt(config.epsilon) * Rt
    L = rv._target_length(Rt, config.epsilon, config.length_pct)
    Rd = config.throat_downstream_radius_factor * Rt
    theta_N, _ = rv._design_angles_rad(
        config.epsilon, config.length_pct, config.thetaN_guess_deg,
    )
    nodes = list(config.kernel_bd)
    bd_x = np.asarray([float(p.x) for p in nodes], dtype=float)
    bd_r = np.asarray([float(p.r) for p in nodes], dtype=float)
    bd_M = np.asarray([float(p.M) for p in nodes], dtype=float)
    bd_th = np.asarray([float(p.theta) for p in nodes], dtype=float)

    return StaticParams(
        n_ce=int(config.n_control),
        n_wall=int(config.n_wall) if config.couple_wall else 0,
        gamma=float(config.gamma),
        physics_weight=float(
            physics_weight if physics_weight is not None else rv.PHYSICS_WEIGHT
        ),
        L_target=float(L),
        Re=float(Re),
        Rt=float(Rt),
        Nx=float(Rd * math.sin(theta_N)),
        Ny=float(Rt + Rd * (1.0 - math.cos(theta_N))),
        mdot_target_throat=float(rv._target_mdot(Rt, config.gamma)),
        bd_full_flux=float(rv.curve_mass_flux(nodes, config.gamma)),
        bd_x=bd_x, bd_r=bd_r, bd_M=bd_M, bd_theta=bd_th,
        active=frozenset(active),
    )


# --------------------------------------------------------------------------- #
# unpack (mirrors _unpack_bvp layout)                                          #
# --------------------------------------------------------------------------- #
def unpack(u, sp: StaticParams):
    """Split the packed unknown vector (layout per ``_pack_bvp``)."""
    n, n_w = sp.n_ce, sp.n_wall
    u = jnp.asarray(u, dtype=jnp.float64)
    M_ce = u[:n]
    th_ce = u[n:2 * n]
    r_ce = u[2 * n:3 * n]
    base = 3 * n
    if n_w > 0:
        w_M = u[base:base + n_w]
        w_th = u[base + n_w:base + 2 * n_w]
        w_x = u[base + 2 * n_w:base + 3 * n_w]
        w_r = u[base + 3 * n_w:base + 4 * n_w]
        s0 = base + 4 * n_w
    else:
        w_M = w_th = w_x = w_r = None
        s0 = base
    lambda2, lambda3, log_C, kdf = u[s0], u[s0 + 1], u[s0 + 2], u[s0 + 3]
    pair_fracs = u[s0 + 4:s0 + 4 + n] if n_w > 0 else None
    return (M_ce, th_ce, r_ce), (w_M, w_th, w_x, w_r), (lambda2, lambda3, log_C, kdf), pair_fracs


# --------------------------------------------------------------------------- #
# kernel BD interpolation + masked partial flux (static polyline, kdf unknown) #
# --------------------------------------------------------------------------- #
def _bd_geometry(sp: StaticParams):
    bx = jnp.asarray(sp.bd_x); br = jnp.asarray(sp.bd_r)
    bM = jnp.asarray(sp.bd_M); bth = jnp.asarray(sp.bd_theta)
    seg = jnp.hypot(bx[1:] - bx[:-1], br[1:] - br[:-1])
    cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(seg)])
    return bx, br, bM, bth, seg, cum


def bd_point_at_fraction(sp: StaticParams, kdf):
    """(x, r, theta) of point D at arc fraction ``kdf`` along BD.

    Equals ``bd_segment_at_fraction(kernel_bd, kdf)[-1]`` (linear state
    interpolation along the polyline arc).
    """
    bx, br, _, bth, _, cum = _bd_geometry(sp)
    total = cum[-1]
    target = jnp.clip(kdf, 0.0, 1.0) * total
    xD = jnp.interp(target, cum, bx)
    rD = jnp.interp(target, cum, br)
    thD = jnp.interp(target, cum, bth)
    return xD, rD, thD


def bd_flux_to_fraction(sp: StaticParams, kdf):
    """Mass flux of the B->D sub-polyline (nasa_moc.calc_mdot_bd).

    Masked partial-segment sum: each segment contributes its flux up to
    the local parameter ``t = clip((target - cum_i)/ds_i, 0, 1)``.  At
    t=1 this is the exact full-segment term of ``nasa_moc.curve_mass_flux``
    (endpoint state = raw node); at 0<t<1 the segment endpoint is the
    linearly interpolated D with the NumPy ``max(M, 1.000001)`` clamp.
    Smooth in ``kdf`` except for the (measure-zero) node-crossing kinks
    the NumPy version has too.
    """
    g = sp.gamma
    bx, br, bM, bth, seg, cum = _bd_geometry(sp)
    total = cum[-1]
    target = jnp.clip(kdf, 0.0, 1.0) * total

    ds_safe = jnp.maximum(seg, 1e-300)
    t = jnp.clip((target - cum[:-1]) / ds_safe, 0.0, 1.0)

    M0, M1 = bM[:-1], bM[1:]
    th0, th1 = bth[:-1], bth[1:]
    r0, r1 = br[:-1], br[1:]
    # Segment endpoint state: raw node at t=1, clamped interpolation else.
    M_end = jnp.where(t >= 1.0, M1, jnp.maximum(M0 + t * (M1 - M0), _NASA_M_FLOOR))
    th_end = th0 + t * (th1 - th0)
    r_end = r0 + t * (r1 - r0)

    dx_f, dr_f = bx[1:] - bx[:-1], br[1:] - br[:-1]
    beta = jnp.arctan2(dr_f, dx_f)            # collinear: same for partial seg
    M_mid = jnp.maximum(0.5 * (M0 + M_end), _NASA_M_FLOOR)
    th_mid = 0.5 * (th0 + th_end)
    r_mid = jnp.maximum(0.5 * (r0 + r_end), 1e-12)
    rho = isentropic_density_ratio(M_mid, g)
    T = isentropic_temperature_ratio(M_mid, g)
    V = M_mid * jnp.sqrt(g * T)
    ds_part = t * seg
    flux = 2.0 * jnp.pi * r_mid * rho * V * jnp.abs(jnp.sin(beta - th_mid)) * ds_part
    return jnp.sum(jnp.where(ds_part > 1e-12, flux, 0.0))


# --------------------------------------------------------------------------- #
# CE geometry reconstruction (mirrors _integrate_ce_x_from_left_mach)          #
# --------------------------------------------------------------------------- #
def integrate_x_from_left_mach(r, theta, M, x_start):
    """x[i+1] = x[i] + dr/tan(θavg+μavg); fallback +1e-9 when |tan|<1e-12."""
    m = jnp.maximum(M, _NASA_M_FLOOR)
    mu = jnp.arcsin(1.0 / m)
    mu_avg = 0.5 * (mu[:-1] + mu[1:])
    th_avg = 0.5 * (theta[:-1] + theta[1:])
    denom = jnp.tan(th_avg + mu_avg)
    dr = r[1:] - r[:-1]
    denom_safe = jnp.where(jnp.abs(denom) < 1e-12, 1.0, denom)
    step = jnp.where(jnp.abs(denom) < 1e-12, 1e-9, dr / denom_safe)
    return x_start + jnp.concatenate([jnp.zeros(1), jnp.cumsum(step)])


# --------------------------------------------------------------------------- #
# flux / leaf helpers (assembly-local, NumPy-exact clamps)                     #
# --------------------------------------------------------------------------- #
def _polyline_mass_flux(x, r, M_node, theta, gamma, m_floor):
    """curve_mass_flux on a polyline whose nodes carry M_node (pre-clamped).

    Midpoint M floor ``m_floor`` (1.001 CE-side, 1.000001 NASA-side),
    midpoint r floor 1e-12, segments with ds<=1e-12 skipped.
    """
    dx = x[1:] - x[:-1]
    dr = r[1:] - r[:-1]
    ds = jnp.hypot(dx, dr)
    ok = ds > 1e-12
    dx_s = jnp.where(ok, dx, 1.0)
    dr_s = jnp.where(ok, dr, 0.0)
    beta = jnp.arctan2(dr_s, dx_s)
    M_mid = jnp.maximum(0.5 * (M_node[:-1] + M_node[1:]), m_floor)
    th_mid = 0.5 * (theta[:-1] + theta[1:])
    r_mid = jnp.maximum(0.5 * (r[:-1] + r[1:]), 1e-12)
    rho = isentropic_density_ratio(M_mid, gamma)
    T = isentropic_temperature_ratio(M_mid, gamma)
    V = M_mid * jnp.sqrt(gamma * T)
    flux = 2.0 * jnp.pi * r_mid * rho * V * jnp.abs(jnp.sin(beta - th_mid)) * ds
    return jnp.sum(jnp.where(ok, flux, 0.0))


def _q_source(theta, mu, r, sign):
    """Anderson Q± with the rao_residuals guards (|cos|<=1e-12, r<=1e-12)."""
    cos_t = jnp.cos(theta + sign * mu)
    ok = (jnp.abs(cos_t) > 1e-12) & (r > 1e-12)
    cos_safe = jnp.where(ok, cos_t, 1.0)
    val = sign * jnp.sin(theta) * jnp.sin(mu) * jnp.cos(mu) / (r * cos_safe)
    return jnp.where(ok, val, 0.0)


def _cplus_pair(x0, r0, M0, th0, mu0, x1, r1, M1, th1, mu1, gamma):
    """rao_residuals.residual_Cplus_axisym on an explicit (p0, p1) pair.

    ``M*`` enter only through ν = PM(max(M, 1.001)); ``mu*`` are the
    *node-stored* Mach angles (CE: asin(1/max(M,1.001)); wall-interp:
    asin(1/max(M,1.000001))) — exactly the FlowNode.mu semantics.
    """
    nu0 = prandtl_meyer(jnp.maximum(M0, _CE_M_FLOOR), gamma)
    nu1 = prandtl_meyer(jnp.maximum(M1, _CE_M_FLOOR), gamma)
    lhs = (th1 + nu1) - (th0 + nu0)
    ds = jnp.hypot(x1 - x0, r1 - r0)
    th = 0.5 * (th0 + th1)
    mu = 0.5 * (mu0 + mu1)
    r = jnp.maximum(0.5 * (r0 + r1), 1e-12)
    return lhs - _q_source(th, mu, r, +1.0) * ds


def _rao_stationarity(M, theta, log_C, gamma):
    """rao_variational.rao_stationarity_residual incl. the singular branch."""
    Mc = jnp.maximum(M, 1.0 + 1e-9)
    alpha = jnp.arcsin(1.0 / Mc)
    cos_a = jnp.cos(alpha)
    cos_tma = jnp.cos(theta - alpha)
    Ms = mstar_from_M(Mc, gamma)
    normal_ok = (cos_a > 1e-12) & (jnp.abs(cos_tma) > 1e-12) & (Ms > 1e-12)
    # double-where: keep log arguments positive on the inactive branch
    safe_tma = jnp.where(normal_ok, jnp.abs(cos_tma), 1.0)
    safe_ca = jnp.where(normal_ok, cos_a, 1.0)
    safe_Ms = jnp.where(normal_ok, Ms, 1.0)
    normal = jnp.log(safe_Ms) + jnp.log(safe_tma) - jnp.log(safe_ca) - log_C
    singular = Ms * cos_tma / jnp.maximum(cos_a, 1e-12) - jnp.exp(log_C)
    return jnp.where(normal_ok, normal, singular)


# --------------------------------------------------------------------------- #
# the assembled residual                                                       #
# --------------------------------------------------------------------------- #
def make_residual(sp: StaticParams):
    """Return the pure flat residual ``fn(u) -> r`` for this configuration.

    Output ordering is ``RaoResidualGroups.flat()``:
    mass | length | algebraic_stationarity | moc_cplus | moc_cminus |
    ce_geometry | regularization | penalties | wall_endpoint |
    wall_tangency | cplus_ce_to_wall | wall_intersection
    (transversality/stationarity/left_mach are zero-length in the default
    block set; inactive blocks contribute zero-length arrays).
    """
    n, n_w = sp.n_ce, sp.n_wall
    g = sp.gamma
    W = sp.physics_weight
    L_t, Re, Rt = sp.L_target, sp.Re, sp.Rt
    x_scale = max(L_t, 1e-12)
    r_scale = max(Re, 1e-12)
    active = sp.active
    coupled = n_w > 0
    empty = jnp.zeros(0, dtype=jnp.float64)

    def residual(u, args=None):
        (M_ce, th_ce, r_ce), (w_M, w_th, w_x, w_r), scalars, pf = unpack(u, sp)
        _, _, log_C, kdf = scalars

        # -- geometry-backed CE (x from left-Mach ODE, start at D) ----------
        xD, rD, thD = bd_point_at_fraction(sp, kdf)
        x_ce = integrate_x_from_left_mach(r_ce, th_ce, M_ce, xD)
        Mn = jnp.maximum(M_ce, _CE_M_FLOOR)      # CE flow-node M (1.001)
        mun = jnp.arcsin(1.0 / Mn)               # FlowNode.mu on CE nodes

        # -- mass closure ----------------------------------------------------
        if "mass" in active:
            ce_flux = _polyline_mass_flux(x_ce, r_ce, Mn, th_ce, g, _CE_M_FLOOR)
            bd_flux = bd_flux_to_fraction(sp, kdf)
            mdot_ref = jnp.maximum(
                jnp.maximum(jnp.abs(sp.bd_full_flux), jnp.abs(bd_flux)),
                jnp.maximum(sp.mdot_target_throat, 1e-12),
            )
            mass = ((ce_flux - bd_flux) / jnp.maximum(mdot_ref, 1e-12))[None]
        else:
            mass = empty

        # -- length (Σ dx over CE segments with ds>1e-12) ---------------------
        if "length" in active:
            dxs = x_ce[1:] - x_ce[:-1]
            dss = jnp.hypot(dxs, r_ce[1:] - r_ce[:-1])
            L_val = jnp.sum(jnp.where(dss >= 1e-12, dxs, 0.0))
            length = ((L_val - L_t) / max(L_t, 1e-12))[None]
        else:
            length = empty

        # -- algebraic Rao stationarity (per node, weight W) ------------------
        if "algebraic_stationarity" in active:
            alg = W * _rao_stationarity(Mn, th_ce, log_C, g)
        else:
            alg = empty

        # -- left-Mach diagnostic (exact by construction; ~1e-12) -------------
        if "left_mach" in active:
            dx = x_ce[1:] - x_ce[:-1]
            dr = r_ce[1:] - r_ce[:-1]
            raw = dr - dx * jnp.tan(0.5 * (th_ce[:-1] + th_ce[1:])
                                    + 0.5 * (mun[:-1] + mun[1:]))
            lm = raw / jnp.maximum(jnp.hypot(dx, dr), 1e-12)
        else:
            lm = empty

        # -- C+/C- compatibility on CE segments (weight W, /1°) ---------------
        if "moc_cplus" in active or "moc_cminus" in active:
            nu = prandtl_meyer(Mn, g)
            ds = jnp.hypot(x_ce[1:] - x_ce[:-1], r_ce[1:] - r_ce[:-1])
            th_avg = 0.5 * (th_ce[:-1] + th_ce[1:])
            mu_avg = 0.5 * (mun[:-1] + mun[1:])
            r_avg = jnp.maximum(0.5 * (r_ce[:-1] + r_ce[1:]), 1e-12)
            kp = th_ce + nu
            km = th_ce - nu
            cp = ((kp[1:] - kp[:-1]) - _q_source(th_avg, mu_avg, r_avg, +1.0) * ds) / _ONE_DEG
            cm = ((km[1:] - km[:-1]) - _q_source(th_avg, mu_avg, r_avg, -1.0) * ds) / _ONE_DEG
            cp = W * cp if "moc_cplus" in active else empty
            cm = W * cm if "moc_cminus" in active else empty
        else:
            cp = cm = empty

        # -- ce_geometry -------------------------------------------------------
        if "ce_geometry" in active:
            start = jnp.stack([
                (r_ce[0] - rD) / r_scale,
                (th_ce[0] - thD) / _ONE_DEG,
            ])
            if coupled:
                endpoint = jnp.stack([
                    (x_ce[-1] - w_x[-1]) / x_scale,
                    (r_ce[-1] - w_r[-1]) / r_scale,
                ])
            else:
                endpoint = jnp.stack([
                    (x_ce[-1] - L_t) / x_scale,
                    (r_ce[-1] - Re) / r_scale,
                ])
            dx = x_ce[1:] - x_ce[:-1]
            dr = r_ce[1:] - r_ce[:-1]
            ce_geom = jnp.concatenate([
                start, endpoint,
                jnp.maximum(-dx, 0.0) / x_scale,
                jnp.maximum(-dr, 0.0) / r_scale,
            ])
        else:
            ce_geom = empty

        # -- smoothness regularization (0.02 × diff², /1°) ----------------------
        if "regularization" in active and n >= 3:
            nu_reg = prandtl_meyer(Mn, g)
            kp = th_ce + nu_reg
            km = th_ce - nu_reg
            reg = 0.02 * jnp.concatenate([
                jnp.diff(kp, n=2), jnp.diff(km, n=2),
            ]) / _ONE_DEG
        else:
            reg = empty

        # -- penalties (raw-M monotonicity; raw pair-fraction monotonicity) -----
        if "penalties" in active:
            mach_pen = jnp.maximum(-(M_ce[1:] - M_ce[:-1]), 0.0) / 0.05
            if coupled:
                pair_pen = jnp.maximum(-(pf[1:] - pf[:-1]), 0.0) / 0.01
                pen = jnp.concatenate([mach_pen, pair_pen])
            else:
                pen = mach_pen
        else:
            pen = empty

        # -- Phase 6 coupled-wall blocks -----------------------------------------
        if coupled:
            # wall arc-length (normalized; degenerate fallback = linspace)
            wseg = jnp.hypot(w_x[1:] - w_x[:-1], w_r[1:] - w_r[:-1])
            warc = jnp.concatenate([jnp.zeros(1), jnp.cumsum(wseg)])
            wtotal = warc[-1]
            arc_norm = jnp.where(
                wtotal > 1e-12, warc / jnp.maximum(wtotal, 1e-300),
                jnp.linspace(0.0, 1.0, n_w),
            )

            if "wall_endpoint" in active:
                wall_ep = jnp.stack([
                    (w_x[0] - sp.Nx) / max(L_t, 1e-12),
                    (w_r[0] - sp.Ny) / max(Re, 1e-12),
                    (w_x[-1] - L_t) / max(L_t, 1e-12),
                    (w_r[-1] - Re) / max(Re, 1e-12),
                ])
            else:
                wall_ep = empty

            if "wall_tangency" in active:
                wall_tan = ((w_r[1:] - w_r[:-1])
                            - (w_x[1:] - w_x[:-1])
                            * jnp.tan(0.5 * (w_th[:-1] + w_th[1:]))) / max(Re, 1e-12)
            else:
                wall_tan = empty

            # free CE↔wall pairing at clipped pair fractions
            f = jnp.clip(pf, 0.0, 1.0)
            wp_x = jnp.interp(f, arc_norm, w_x)
            wp_r = jnp.interp(f, arc_norm, w_r)
            wp_M = jnp.maximum(jnp.interp(f, arc_norm, w_M), _NASA_M_FLOOR)
            wp_th = jnp.interp(f, arc_norm, w_th)
            wp_mu = jnp.arcsin(1.0 / wp_M)        # FlowNode.mu (1.000001 floor)

            if "cplus_ce_to_wall" in active:
                cw = W * _cplus_pair(
                    x_ce, r_ce, Mn, th_ce, mun,
                    wp_x, wp_r, wp_M, wp_th, wp_mu, g,
                ) / _ONE_DEG
            else:
                cw = empty

            if "wall_intersection" in active:
                th_avg = 0.5 * (th_ce + wp_th)
                mu_avg = 0.5 * (mun + wp_mu)
                wi = ((wp_r - r_ce) - jnp.tan(th_avg + mu_avg) * (wp_x - x_ce)) \
                    / max(Re, 1e-12)
            else:
                wi = empty
        else:
            wall_ep = wall_tan = cw = wi = empty

        # RaoResidualGroups.flat() order (transversality/stationarity empty)
        return jnp.concatenate([
            mass, length, alg, lm, cp, cm, ce_geom, reg, pen,
            wall_ep, wall_tan, cw, wi,
        ])

    return residual


__all__ = [
    "StaticParams", "params_from_config", "make_residual", "unpack",
    "bd_point_at_fraction", "bd_flux_to_fraction",
    "integrate_x_from_left_mach", "SUPPORTED_BLOCKS",
]
