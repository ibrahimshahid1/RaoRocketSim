"""
raosim.jax.moc_kernel — differentiable throat-arc kernel march (J3b).

Plan reference: JAX_DIFFERENTIABLE_PLAN.md §4.2 / §5.  Ports the
*Anderson-form* axisymmetric unit processes from ``raosim.moc`` —
``solve_interior_point`` / ``solve_axis_point`` / ``solve_wall_point``
with the CORRECTED characteristic pairing (2026-06-11; Anderson MCF
§11.4, Zucrow & Hoffman Vol. 2 Ch. 17, oracle-proven in
``tests/test_characteristic_pairing.py``)::

    along C− (slope θ−μ):  d(θ+ν) = +S ds
    along C+ (slope θ+μ):  d(θ−ν) = −S ds,      S = sinθ sinμ / r

This module reaches BIT-PARITY with the NumPy oracle ``build_kernel``
(M3.5Perf-validated) at the BD level — it ports NASA's *exact* row
policy and dθ-form unit processes (below), not merely the same physics.
The Anderson-form processes (``interior_point`` / ``axis_point`` /
``wall_point_arc``) are kept as tested reference ports of the corrected
``raosim.moc`` unit processes, but the MARCH uses the dθ-form ports
(``*_nasa``) because BD-oracle agreement requires NASA's near-axis
coefficient treatment (the Anderson midpoint-source chain converges to
a *different* continuum near the axis: axis Mach +9%, resolution-
divergent).

March structure (the J3b θ_B-differentiability design)
------------------------------------------------------
The throat expansion fan keeps emitting new C+ characteristics at the
wall corner, so the kernel mesh genuinely GROWS toward BD.  NASA limits
the wall step to ``dtheta_limit`` by MIXING two row types
(``_calc_arc_wall_point_with_special``):

* RAW rows (~5%: marched-row indices 2/9/21/39/65 at the reference) —
  the C+ leaving prev[1] TERMINATES on the throat arc and becomes the
  new wall point (every C+ in the fan ends on the wall; this is
  physics, not bookkeeping).  Width is unchanged; the interior C+
  pairing shifts by one (``ii = i + 1``).
* SPECIAL rows (~95%) — when the raw step would exceed ``dtheta_limit``
  (or overshoot θ_B) a prescribed-angle wall point is inserted at
  ``min(θ_B, θ_prev + dtheta_limit/2)``; the row GROWS by one and the
  pairing is same-index (``ii = i``).

Skipping the raw rows (an all-special march) starves the wall-adjacent
strip — measured ν_w − θ_w → −0.67° at B where the oracle's is +0.68°
(axisymmetric wall expansion must run AHEAD of planar PM).  Two other
static-shape designs failed earlier: a static-width forward march
blows up (no step limit; wall steps 0.56°→2.3°→5.4°→NaN); a regridded
inverse march is weakly unstable (axis Mach runaway).

The landed march carries NASA's policy in static PADDED shapes
(``W = n0 + n_rows``): the raw/special decision, valid row width
(``axis_idx``), and pairing offset are TRACED; padded slots hold the
row's axis state (benign finite sentinels so masked lanes never
produce NaNs — the §5 where-grad trap).  Marching freezes once the
wall reaches θ_B; the scan carry at the end IS the BD row.

θ_B-differentiability: the pre-clamp wall grid (raw landings + fixed
``dtheta_limit`` special steps) is θ_B-INDEPENDENT, so
``d(BD)/d(theta_B)`` flows only through the final clamped row — exact
and smooth within a row-count window; the window boundary (θ_B
crossing a grid step) is handled by per-rung re-assembly exactly like
``kernel_bd`` re-seeding today.  ``n_rows`` is a static upper bound on
the row count (frozen rows no-op).

The transonic start line is taken as INPUT ARRAYS (wall-first), e.g.
from the NumPy seed kernel's TT' row: the start line is
θ_B-independent, so ``d/d(theta_B)`` is exact with a frozen start line.
(Differentiable Kliegel–Levine for d/d(Rt, γ) totals is the J3b-2 /
args-lifting increment, tracked in the plan.)

Unit-process parity contract: the Anderson-form ports at ``n_corr=10``
reproduce the NumPy ``raosim.moc`` fixed point (early break off) to
float64 round-off; the dθ-form ``*_nasa`` ports match the NumPy NASA
processes at their 1e-10 fixed points — both pinned in
``tests/test_jax_moc_kernel.py``.
"""

from __future__ import annotations

from typing import NamedTuple

import raosim.jax  # noqa: F401  (enables x64 before any array is built)

import jax
import jax.numpy as jnp
from jax import lax

from raosim.jax.primitives import (
    mach_angle,
    mach_from_prandtl_meyer,
    prandtl_meyer,
)

_NU_FLOOR = 1e-8


class KernelRow(NamedTuple):
    """One characteristic row, wall-first (index 0 = wall)."""

    x: jax.Array
    r: jax.Array
    theta: jax.Array
    nu: jax.Array
    mu: jax.Array
    M: jax.Array


class KernelMarch(NamedTuple):
    """March result.

    ``rows`` stacks the padded start line + all scan rows (shape
    ``(n_rows + 1, n0 + n_rows)`` per field, wall-first; row k's valid
    prefix is ``row_axis_idx[k] + 1`` nodes, the rest is axis-state
    padding; rows after the march froze repeat BD).  ``bd`` is the
    final row — the kernel's BD at exactly ``theta_B`` — valid through
    index ``bd_axis_idx`` (traced: the raw/special row mix is
    data-dependent).  ``reached_wall`` mirrors the NumPy kernel flag
    (False means ``n_rows`` was too small a bound)."""

    rows: KernelRow
    bd: KernelRow
    bd_axis_idx: jax.Array     # axis index of BD (valid count − 1)
    reached_wall: jax.Array    # bool: wall reached theta_B in n_rows
    row_axis_idx: jax.Array    # (n_rows + 1,) per-row axis index
    wall_x: jax.Array          # (n_rows + 1,) wall-point trace
    wall_r: jax.Array
    wall_theta: jax.Array


def _row_point(row: KernelRow, i):
    """Scalar point pytree for node ``i`` of a row (static or traced i)."""
    return dict(
        x=row.x[i], r=row.r[i], theta=row.theta[i],
        nu=row.nu[i], mu=row.mu[i], M=row.M[i],
    )


# --------------------------------------------------------------------------- #
# Unit processes (fixed-iteration ports of raosim.moc, corrected pairing)     #
# --------------------------------------------------------------------------- #
def interior_point(pm: dict, pp: dict, gamma, n_corr: int = 10) -> dict:
    """Interior point: C− from ``pm`` (above) ∩ C+ from ``pp`` (below).

    Port of ``moc.solve_interior_point`` with the early ``tol`` break
    removed (fixed ``n_corr`` corrector passes; extra passes are no-ops
    at the fixed point).  All NumPy guards become ``jnp.where``.
    """
    km0 = pm["theta"] + pm["nu"]   # (θ+ν) carried along C−
    kp0 = pp["theta"] - pp["nu"]   # (θ−ν) carried along C+

    theta3 = 0.5 * (km0 + kp0)
    nu3 = jnp.maximum(0.5 * (km0 - kp0), _NU_FLOOR)
    M3 = mach_from_prandtl_meyer(nu3, gamma)
    mu3 = mach_angle(M3)
    x3 = 0.5 * (pm["x"] + pp["x"])
    r3 = 0.5 * (pm["r"] + pp["r"])

    def body(_, carry):
        x3, r3, theta3, nu3, mu3, M3 = carry
        slope_m = jnp.tan(0.5 * (pm["theta"] + theta3)
                          - 0.5 * (pm["mu"] + mu3))
        slope_p = jnp.tan(0.5 * (pp["theta"] + theta3)
                          + 0.5 * (pp["mu"] + mu3))
        denom = slope_m - slope_p
        ok = jnp.abs(denom) > 1e-15
        denom_safe = jnp.where(ok, denom, 1.0)
        x3_new = ((pp["r"] - pm["r"]) - slope_p * pp["x"]
                  + slope_m * pm["x"]) / denom_safe
        r3_new = pm["r"] + slope_m * (x3_new - pm["x"])
        x3 = jnp.where(ok, x3_new, x3)
        r3 = jnp.where(ok, r3_new, r3)
        r3 = jnp.maximum(r3, 0.0)

        # Axisymmetric source along each family (midpoint rule).
        ds_m = jnp.sqrt((x3 - pm["x"]) ** 2 + (r3 - pm["r"]) ** 2)
        ds_p = jnp.sqrt((x3 - pp["x"]) ** 2 + (r3 - pp["r"]) ** 2)
        th_m = 0.5 * (pm["theta"] + theta3)
        mu_m = 0.5 * (pm["mu"] + mu3)
        r_m = jnp.maximum(0.5 * (pm["r"] + r3), 1e-12)
        th_p = 0.5 * (pp["theta"] + theta3)
        mu_p = 0.5 * (pp["mu"] + mu3)
        r_p = jnp.maximum(0.5 * (pp["r"] + r3), 1e-12)
        S_m = jnp.sin(th_m) * jnp.sin(mu_m) / r_m
        S_p = jnp.sin(th_p) * jnp.sin(mu_p) / r_p
        src = r3 > 1e-10
        k_minus = jnp.where(src, km0 + S_m * ds_m, km0)
        k_plus = jnp.where(src, kp0 - S_p * ds_p, kp0)

        theta3 = 0.5 * (k_minus + k_plus)
        nu3 = jnp.maximum(0.5 * (k_minus - k_plus), _NU_FLOOR)
        M3 = mach_from_prandtl_meyer(nu3, gamma)
        mu3 = mach_angle(M3)
        return (x3, r3, theta3, nu3, mu3, M3)

    x3, r3, theta3, nu3, mu3, M3 = lax.fori_loop(
        0, n_corr, body, (x3, r3, theta3, nu3, mu3, M3)
    )

    # Downstream-progression guard (degenerate configurations).
    x_min_parent = jnp.minimum(pm["x"], pp["x"])
    bad = x3 < x_min_parent
    x3 = jnp.where(
        bad,
        x_min_parent + 1e-8 * jnp.maximum(jnp.abs(x_min_parent), 1e-6),
        x3,
    )
    r3 = jnp.where(bad, 0.5 * (pm["r"] + pp["r"]), r3)
    return dict(x=x3, r=r3, theta=theta3, nu=nu3, mu=mu3, M=M3)


def axis_point(pa: dict, gamma, n_corr: int = 10) -> dict:
    """Axis closure: C− from ``pa`` reaches r=0, θ=0 by symmetry.

    Port of ``moc.solve_axis_point`` (which already runs a fixed
    ``max_iter`` loop with no early break — exact mirror).
    """
    km0 = pa["theta"] + pa["nu"]
    nu3 = jnp.maximum(km0, _NU_FLOOR)
    M3 = mach_from_prandtl_meyer(nu3, gamma)
    mu3 = mach_angle(M3)
    x3 = pa["x"]

    def body(_, carry):
        x3, nu3, mu3, M3 = carry
        slope_m = jnp.tan(0.5 * pa["theta"] - 0.5 * (pa["mu"] + mu3))
        ok = jnp.abs(slope_m) > 1e-15
        x3 = jnp.where(
            ok,
            pa["x"] - pa["r"] / jnp.where(ok, slope_m, 1.0),
            pa["x"] + 2.0 * pa["r"],
        )
        ds = jnp.sqrt((x3 - pa["x"]) ** 2 + pa["r"] ** 2)
        th_avg = 0.5 * pa["theta"]
        mu_avg = 0.5 * (pa["mu"] + mu3)
        r_avg = jnp.maximum(0.5 * pa["r"], 1e-12)
        # Small-angle substitution sin(θ)≈θ near the axis (NumPy parity).
        sin_th = jnp.where(jnp.abs(th_avg) < 1e-10, th_avg, jnp.sin(th_avg))
        S = sin_th * jnp.sin(mu_avg) / r_avg
        k_minus = jnp.where(pa["r"] > 1e-10, km0 + S * ds, km0)
        nu3 = jnp.maximum(k_minus, _NU_FLOOR)
        M3 = mach_from_prandtl_meyer(nu3, gamma)
        mu3 = mach_angle(M3)
        return (x3, nu3, mu3, M3)

    x3, nu3, mu3, M3 = lax.fori_loop(0, n_corr, body, (x3, nu3, mu3, M3))
    zero = jnp.zeros_like(x3)
    return dict(x=x3, r=zero, theta=zero, nu=nu3, mu=mu3, M=M3)


def _arc_intersect(x0, r0, slope, Rt, Rd):
    """Closed-form C+ ray ∩ throat circle (centre (0, Rt+Rd), radius Rd).

    Replaces ``nasa_moc.ArcWall.intersect_char``'s clamped Newton with
    the quadratic root on the lower half of the circle nearest the ray
    origin (the first crossing).  Differentiable; the discriminant is
    floored at 0.
    """
    c = Rt + Rd
    b = r0 - slope * x0 - c
    a = 1.0 + slope * slope
    disc = jnp.maximum(slope * slope * b * b - a * (b * b - Rd * Rd), 0.0)
    x_hit = (-slope * b - jnp.sqrt(disc)) / a
    x_hit = jnp.clip(x_hit, 0.0, Rd)
    r_hit = c - jnp.sqrt(jnp.maximum(Rd * Rd - x_hit * x_hit, 0.0))
    return x_hit, r_hit


def wall_point_arc(pi: dict, Rt, Rd, gamma, n_corr: int = 10) -> dict:
    """Forward arc wall point: C+ from ``pi`` (inside) hits the circle.

    Port of ``moc.solve_wall_point`` specialised to the throat arc:
    the Newton ``intersect_char`` becomes the closed-form quadratic and
    ``wall.theta(x) = asin(x/Rd)`` (the arc is parametrised so the flow
    angle equals the arc angle).  NOT used by ``march_kernel`` (the
    free landing angle is what blows up a static-width march — see the
    module docstring); kept as the tested port of the forward process
    for diagnostics and future couplings.
    """
    kp0 = pi["theta"] - pi["nu"]

    # Predictor (slopes from the inside point alone), as in NumPy.
    slope = jnp.tan(pi["theta"] + pi["mu"])
    x_hit, r_hit = _arc_intersect(pi["x"], pi["r"], slope, Rt, Rd)
    theta_w = jnp.arcsin(jnp.clip(x_hit / Rd, 0.0, 1.0))

    ds = jnp.sqrt((x_hit - pi["x"]) ** 2 + (r_hit - pi["r"]) ** 2)
    th_avg = 0.5 * (pi["theta"] + theta_w)
    r_avg = jnp.maximum(0.5 * (pi["r"] + r_hit), 1e-12)
    S = jnp.sin(th_avg) * jnp.sin(pi["mu"]) / r_avg
    src = (r_hit > 1e-10) & (pi["r"] > 1e-10)
    k_plus = jnp.where(src, kp0 - S * ds, kp0)
    nu_w = jnp.maximum(theta_w - k_plus, _NU_FLOOR)
    M_w = mach_from_prandtl_meyer(nu_w, gamma)
    mu_w = mach_angle(M_w)

    def body(_, carry):
        x_hit, r_hit, theta_w, nu_w, mu_w, M_w = carry
        slope = jnp.tan(0.5 * (pi["theta"] + theta_w)
                        + 0.5 * (pi["mu"] + mu_w))
        x_hit, r_hit = _arc_intersect(pi["x"], pi["r"], slope, Rt, Rd)
        theta_w = jnp.arcsin(jnp.clip(x_hit / Rd, 0.0, 1.0))
        ds = jnp.sqrt((x_hit - pi["x"]) ** 2 + (r_hit - pi["r"]) ** 2)
        th_avg = 0.5 * (pi["theta"] + theta_w)
        mu_avg = 0.5 * (pi["mu"] + mu_w)
        r_avg = jnp.maximum(0.5 * (pi["r"] + r_hit), 1e-12)
        S = jnp.sin(th_avg) * jnp.sin(mu_avg) / r_avg
        src = (r_hit > 1e-10) & (pi["r"] > 1e-10)
        k_plus = jnp.where(src, kp0 - S * ds, kp0)
        nu_w = jnp.maximum(theta_w - k_plus, _NU_FLOOR)
        M_w = mach_from_prandtl_meyer(nu_w, gamma)
        mu_w = mach_angle(M_w)
        return (x_hit, r_hit, theta_w, nu_w, mu_w, M_w)

    x_hit, r_hit, theta_w, nu_w, mu_w, M_w = lax.fori_loop(
        0, n_corr, body, (x_hit, r_hit, theta_w, nu_w, mu_w, M_w)
    )
    return dict(x=x_hit, r=r_hit, theta=theta_w, nu=nu_w, mu=mu_w, M=M_w)


# --------------------------------------------------------------------------- #
# NASA dθ-form unit processes (Deriv-shaped; used by the march)               #
#                                                                             #
# Exact ports of nasa_moc._calc_interior_mesh_point /                         #
# _calc_axial_mesh_point / _calc_special_wall_point with fixed corrector      #
# passes (the NumPy loops iterate a contracting fixed point to               #
# conv_tol=1e-10; a fixed pass count lands on the same fixed point — parity   #
# pinned at 1e-8 in tests).  These forms work directly in M (no              #
# Prandtl–Meyer inversion inside the correctors) and integrate the           #
# axisymmetric source via the conditioned z-form/r-form coefficient pairs    #
# (B, b, R, R*) evaluated at the segment endpoints — measured necessary for  #
# BD oracle agreement: the Anderson midpoint-source chain converges to a     #
# *different* continuum near the axis (axis Mach +9% and                     #
# resolution-divergent; see the module docstring history).                   #
# --------------------------------------------------------------------------- #
def _mm(M):
    """``MM = sqrt(M²−1)`` (NASA C++ line 3046), sonic-floored."""
    return jnp.sqrt(jnp.maximum(M * M - 1.0, 1e-30))


def _coef_A(M, gamma):
    return _mm(M) / (M * (1.0 + 0.5 * (gamma - 1.0) * M * M))


def _guarded_inv(denom, bad):
    """``where(bad, 0, 1/denom)`` without the where-NaN-grad trap."""
    bad = bad | (jnp.abs(denom) < 1e-12)
    return jnp.where(bad, 0.0, 1.0 / jnp.where(bad, 1.0, denom))


def _coef_B(M, theta, r):
    """z-form LRC source coefficient: 1/(r·(MM/tanθ − 1)); 0 on axis."""
    bad = (r == 0.0) | (jnp.abs(theta) < 1e-9)
    tan_safe = jnp.tan(jnp.where(bad, 1.0, theta))
    return _guarded_inv(r * (_mm(M) / tan_safe - 1.0), bad)


def _coef_b(M, theta, r):
    """z-form RRC source coefficient: 1/(r·(MM/tanθ + 1)); 0 on axis."""
    bad = (r == 0.0) | (jnp.abs(theta) < 1e-9)
    tan_safe = jnp.tan(jnp.where(bad, 1.0, theta))
    return _guarded_inv(r * (_mm(M) / tan_safe + 1.0), bad)


def _coef_R(M, theta, r):
    """r-form LRC source coefficient: 1/(r·(MM + cotθ)); 0 on axis."""
    bad = (r == 0.0) | (jnp.abs(theta) < 1e-9)
    tan_safe = jnp.tan(jnp.where(bad, 1.0, theta))
    return _guarded_inv(r * (_mm(M) + 1.0 / tan_safe), bad)


def _coef_Rs(M, theta, r):
    """r-form RRC source coefficient: 1/(r·(MM − cotθ)); 0 on axis."""
    bad = (r == 0.0) | (jnp.abs(theta) < 1e-9)
    tan_safe = jnp.tan(jnp.where(bad, 1.0, theta))
    return _guarded_inv(r * (_mm(M) - 1.0 / tan_safe), bad)


def _tan_avg(x, y):
    """``tan(½(atan x + atan y))`` — NASA C++ line 3037."""
    return jnp.tan(0.5 * (jnp.arctan(x) + jnp.arctan(y)))


def _mu_of(M):
    return jnp.arcsin(jnp.clip(1.0 / jnp.maximum(M, 1.000001), 0.0, 1.0))


def interior_point_nasa(
    p1: dict, p1_off: dict, p2: dict, gamma, n_corr: int = 40,
) -> dict:
    """NASA ``CalcInteriorMeshPoints`` point (C++ 2466) — exact port.

    ``p1`` is the previous row's C+ source node, ``p2`` the new row's
    just-completed neighbour (C− source).  ``p1_off`` is the node above
    ``p1`` on the previous row: when ``p1`` sits ON the axis NASA
    evaluates the C+ source coefficients (B1, R1) at this off-axis
    neighbour instead (the sinθ/r → dθ/dr limit, one-sided) — the
    near-axis treatment the Anderson midpoint form got measurably
    wrong.  Fixed ``n_corr`` passes (NumPy: conv_tol=1e-10).
    """
    M1 = jnp.maximum(p1["M"], 1.000001)
    M2 = jnp.maximum(p2["M"], 1.000001)
    TH1, TH2 = p1["theta"], p2["theta"]
    s1 = jnp.tan(TH1 + _mu_of(M1))          # LRC slope at p1
    s2 = jnp.tan(TH2 - _mu_of(M2))          # RRC slope at p2
    A1 = _coef_A(M1, gamma)
    A2 = _coef_A(M2, gamma)

    on_axis = p1["r"] == 0.0
    M1o = jnp.maximum(p1_off["M"], 1.000001)
    B1 = jnp.where(on_axis,
                   _coef_B(M1o, p1_off["theta"], p1_off["r"]),
                   _coef_B(M1, TH1, p1["r"]))
    R1 = jnp.where(on_axis,
                   _coef_R(M1o, p1_off["theta"], p1_off["r"]),
                   _coef_R(M1, TH1, p1["r"]))

    B2 = _coef_B(M2, TH2, p2["r"])
    b2 = _coef_b(M2, TH2, p2["r"])
    R2 = _coef_R(M2, TH2, p2["r"])
    RS2 = _coef_Rs(M2, TH2, p2["r"])

    init = (
        s1, s2,                    # s3lrc, s3rrc
        b2, B1, R1, RS2,           # b3, B3, R3, RS3
        0.5 * (A1 + A2),           # A3
        TH1,                       # TH3
        0.5 * (p1["x"] + p2["x"]),  # x3 (placeholder; first pass solves)
        0.5 * (p1["r"] + p2["r"]),  # r3
        M2,                        # M3 (placeholder)
    )

    def body(_, carry):
        s3lrc, s3rrc, b3, B3, R3, RS3, A3, TH3, x3, r3, M3 = carry
        slope13 = _tan_avg(s1, s3lrc)
        slope23 = _tan_avg(s2, s3rrc)
        denom = slope23 - slope13
        ok = jnp.abs(denom) > 1e-14
        x3_solve = ((p1["r"] - p2["r"] - slope13 * p1["x"]
                     + slope23 * p2["x"]) / jnp.where(ok, denom, 1.0))
        x3 = jnp.where(
            slope13 > 10000.0, p2["x"],
            jnp.where(slope23 > 10000.0, p1["x"],
                      jnp.where(ok, x3_solve, x3)),
        )
        r3 = jnp.where(
            jnp.abs(s2) <= jnp.abs(s1),
            p2["r"] + slope23 * (x3 - p2["x"]),
            p1["r"] + slope13 * (x3 - p1["x"]),
        )
        T2 = jnp.where(
            jnp.abs(b2) <= jnp.abs(RS2),
            (x3 - p2["x"]) * (b2 + b3),
            (r3 - p2["r"]) * (RS3 + RS2),
        )
        T1 = jnp.where(
            jnp.abs(B1) <= jnp.abs(R1),
            (x3 - p1["x"]) * (B1 + B3),
            (r3 - p1["r"]) * (R3 + R1),
        )
        denom_m = A1 + A2 + 2.0 * A3
        M3 = ((2.0 * (TH2 - TH1) + M2 * (A2 + A3) + M1 * (A1 + A3)
               + T1 + T2) / jnp.where(jnp.abs(denom_m) > 1e-14,
                                      denom_m, 1.0))
        M3 = jnp.maximum(M3, 1.000001)
        A3 = _coef_A(M3, gamma)
        TH3 = (0.5 * (TH1 + TH2)
               + 0.25 * (M2 * (A3 + A2) - M1 * (A1 + A3)
                         - M3 * (A2 - A1) + T2 - T1))
        TH3 = jnp.maximum(TH3, 0.0)
        mu3 = _mu_of(M3)
        s3lrc = jnp.tan(TH3 + mu3)
        s3rrc = jnp.tan(TH3 - mu3)
        B3 = _coef_B(M3, TH3, r3)
        b3 = _coef_b(M3, TH3, r3)
        R3 = _coef_R(M3, TH3, r3)
        RS3 = _coef_Rs(M3, TH3, r3)
        return (s3lrc, s3rrc, b3, B3, R3, RS3, A3, TH3, x3, r3, M3)

    out = lax.fori_loop(0, n_corr, body, init)
    TH3, x3, r3, M3 = out[7], out[8], out[9], out[10]
    r3 = jnp.maximum(r3, 0.0)
    nu3 = jnp.maximum(prandtl_meyer(M3, gamma), _NU_FLOOR)
    return dict(x=x3, r=r3, theta=TH3, nu=nu3, mu=_mu_of(M3), M=M3)


def axis_point_nasa(p2: dict, gamma, n_corr: int = 40) -> dict:
    """NASA ``CalcAxialMeshPoint`` (C++ 2262) — exact port.

    One-sided source: ``M3 = M2 + 2(θ2 + b2·(x3−x2))/(A2+A3)`` with the
    RRC z-form coefficient ``b2`` evaluated at the off-axis endpoint.
    """
    M2 = jnp.maximum(p2["M"], 1.000001)
    TH2 = p2["theta"]
    s2 = jnp.tan(TH2 - _mu_of(M2))
    A2 = _coef_A(M2, gamma)
    b2 = _coef_b(M2, TH2, p2["r"])

    def body(_, carry):
        s3, A3, x3, M3 = carry
        slope23 = _tan_avg(s2, s3)
        ok = jnp.abs(slope23) > 1e-14
        x3 = jnp.where(ok, p2["x"] - p2["r"] / jnp.where(ok, slope23, 1.0),
                       x3)
        denom = A2 + A3
        M3 = M2 + 2.0 * (TH2 + b2 * (x3 - p2["x"])) / jnp.where(
            jnp.abs(denom) > 1e-14, denom, 1.0)
        M3 = jnp.maximum(M3, 1.000001)
        s3 = jnp.tan(-_mu_of(M3))
        A3 = _coef_A(M3, gamma)
        return (s3, A3, x3, M3)

    _, _, x3, M3 = lax.fori_loop(0, n_corr, body, (s2, A2, p2["x"], M2))
    zero = jnp.zeros_like(x3)
    nu3 = jnp.maximum(prandtl_meyer(M3, gamma), _NU_FLOOR)
    return dict(x=x3, r=zero, theta=zero, nu=nu3, mu=_mu_of(M3), M=M3)


def arc_wall_point_raw_nasa(
    prev: KernelRow, Rt, Rd, gamma, n_corr: int = 40,
) -> dict:
    """NASA ``CalcArcWallPoint`` (C++ 835-948) — exact port.

    The C+ leaving prev[1] (the previous row's next-to-wall node) is
    intersected with the throat circle; that C+ line TERMINATES at the
    wall and becomes the new wall point.  Termination is physical —
    every C+ in the fan ends on the wall — and skipping it (an
    all-special march) starves the wall-adjacent strip: measured
    ν_w − θ_w drifts to −0.67° at B where the oracle's is +0.68°
    (axisymmetric wall expansion must run AHEAD of planar PM).
    """
    x1, r1 = prev.x[1], prev.r[1]
    TH1 = prev.theta[1]
    M1 = jnp.maximum(prev.M[1], 1.000001)
    mu1 = _mu_of(M1)
    r_prev_wall = prev.r[0]

    slrc1 = jnp.tan(TH1 + mu1)
    A1 = _coef_A(M1, gamma)
    B1 = _coef_B(M1, TH1, r1)
    R1 = _coef_R(M1, TH1, r1)

    c = Rt + Rd

    def body(_, carry):
        slrc3, A3, B3, R3, x3, r3, theta3, M3 = carry
        slrc13 = _tan_avg(slrc1, slrc3)
        ok = jnp.abs(slrc13) > 1e-14
        x3 = jnp.where(ok, (r3 - r1) / jnp.where(ok, slrc13, 1.0) + x1, x3)
        x3 = jnp.clip(x3, 0.0, Rd)
        inside = jnp.maximum(Rd * Rd - x3 * x3, 0.0)
        r3 = c - jnp.sqrt(inside)
        theta3 = jnp.arcsin(jnp.clip(x3 / Rd, -1.0, 1.0))

        # NOTE: NASA compares B1 <= R1 SIGNED here (C++ line 954), not
        # by magnitude as in the interior process — ported as-is.
        T1 = jnp.where(
            B1 <= R1,
            (x3 - x1) * (B3 + B1),
            (r3 - r1) * (R3 + R1),
        )
        denom = 0.5 * (A1 + A3)
        M3 = M1 + (theta3 - TH1 + 0.5 * T1) / jnp.where(
            jnp.abs(denom) > 1e-14, denom, 1.0)
        M3 = jnp.maximum(M3, 1.000001)

        slrc3 = jnp.tan(theta3 + _mu_of(M3))
        A3 = _coef_A(M3, gamma)
        B3 = _coef_B(M3, theta3, r3)
        R3 = _coef_R(M3, theta3, r3)
        return (slrc3, A3, B3, R3, x3, r3, theta3, M3)

    init = (slrc1, A1, B1, R1, x1, r_prev_wall, TH1, M1)
    out = lax.fori_loop(0, n_corr, body, init)
    x3, r3, theta3, M3 = out[4], out[5], out[6], out[7]
    nu3 = jnp.maximum(prandtl_meyer(M3, gamma), _NU_FLOOR)
    return dict(x=x3, r=r3, theta=theta3, nu=nu3, mu=_mu_of(M3), M=M3)


def special_wall_point_nasa(
    prev: KernelRow, alpha, Rt, Rd, gamma, n_corr: int = 40,
) -> dict:
    """NASA ``CalcSpecialWallPoint`` (C++ — small arc-angle increments)
    — exact port at the prescribed arc angle ``alpha``.

    The C+ foot (point 4) is the intersection of the previous row's
    averaged RRC direction line through prev[0] with the connector's
    averaged LRC line through W; foot states interpolate linearly in x
    between prev[1] and prev[0] (``ratio``, guarded → 0 on x-degenerate
    segments, i.e. the foot takes prev[1]'s state on TT').  Slopes
    interpolate in tan space, exactly as the oracle does.
    """
    x1, x2 = prev.x[1], prev.x[0]
    r1, r2 = prev.r[1], prev.r[0]
    TH1, TH2 = prev.theta[1], prev.theta[0]
    M1 = jnp.maximum(prev.M[1], 1.000001)
    M2 = jnp.maximum(prev.M[0], 1.000001)
    mu1, mu2 = _mu_of(M1), _mu_of(M2)

    theta3 = alpha
    x3 = Rd * jnp.sin(theta3)
    r3 = (Rt + Rd) - Rd * jnp.cos(theta3)

    slrc1 = jnp.tan(TH1 + mu1)
    slrc2 = jnp.tan(TH2 + mu2)
    srrc1 = jnp.tan(TH1 - mu1)
    srrc2 = jnp.tan(TH2 - mu2)
    A1 = _coef_A(M1, gamma)
    A2 = _coef_A(M2, gamma)
    B1 = _coef_B(M1, TH1, r1)
    B2 = _coef_B(M2, TH2, r2)
    R1 = _coef_R(M1, TH1, r1)
    R2 = _coef_R(M2, TH2, r2)
    s4rrc = _tan_avg(srrc1, srrc2)

    def body(_, carry):
        slrc3, slrc4, A3, B3, R3, M3 = carry
        slope34 = _tan_avg(slrc3, slrc4)
        denom = s4rrc - slope34
        degenerate = jnp.abs(denom) < 1e-14
        steep = jnp.abs(slope34) >= 10000.0
        x4_solve = ((r3 - r2 + s4rrc * x2 - slope34 * x3)
                    / jnp.where(degenerate, 1.0, denom))
        x4 = jnp.where(degenerate | steep, x3, x4_solve)
        seg = x2 - x1
        ok = jnp.abs(seg) > 1e-14
        ratio = jnp.where(ok, (x4 - x1) / jnp.where(ok, seg, 1.0), 0.0)

        A4 = A1 + ratio * (A2 - A1)
        theta4 = TH1 + ratio * (TH2 - TH1)
        slrc4 = slrc1 + ratio * (slrc2 - slrc1)
        M4 = M1 + ratio * (M2 - M1)

        B4 = B1 + ratio * (B2 - B1)
        R4 = R1 + ratio * (R2 - R1)
        r4 = r1 + ratio * (r2 - r1)
        T4 = jnp.where(
            jnp.abs(B2) <= jnp.abs(R2),
            (x3 - x4) * (B3 + B4),
            (r3 - r4) * (R3 + R4),
        )
        denom_m = 0.5 * (A4 + A3)
        M3 = M4 + (theta3 - theta4 + 0.5 * T4) / jnp.where(
            jnp.abs(denom_m) > 1e-14, denom_m, 1.0)
        M3 = jnp.maximum(M3, 1.000001)

        slrc3 = jnp.tan(theta3 + _mu_of(M3))
        A3 = _coef_A(M3, gamma)
        B3 = _coef_B(M3, theta3, r3)
        R3 = _coef_R(M3, theta3, r3)
        return (slrc3, slrc4, A3, B3, R3, M3)

    out = lax.fori_loop(0, n_corr, body, (slrc1, slrc2, A1, B1, R1, M1))
    M3 = out[5]
    nu3 = jnp.maximum(prandtl_meyer(M3, gamma), _NU_FLOOR)
    return dict(x=x3, r=r3, theta=theta3, nu=nu3, mu=_mu_of(M3), M=M3)


# --------------------------------------------------------------------------- #
# Padded row sweep + march                                                     #
# --------------------------------------------------------------------------- #
def _set_node(row: KernelRow, i, p: dict) -> KernelRow:
    return KernelRow(
        x=row.x.at[i].set(p["x"]), r=row.r.at[i].set(p["r"]),
        theta=row.theta.at[i].set(p["theta"]),
        nu=row.nu.at[i].set(p["nu"]),
        mu=row.mu.at[i].set(p["mu"]), M=row.M.at[i].set(p["M"]),
    )


def _sweep_row_padded(
    wall_pt: dict, prev: KernelRow, axis_idx, shift, gamma,
    n_corr: int = 40,
) -> KernelRow:
    """Fill a padded row: interior i = 1..axis_idx−1 from
    (new[i−1] C−, prev[i+shift] C+), then the axis closure at
    ``axis_idx``; padded slots beyond it are filled with the axis state
    (benign finite sentinels for masked lanes).

    ``shift`` is NASA's pairing offset (``ii = i if special else
    i+1``): 0 for special rows (the row GREW by the inserted wall
    point; every previous C+ continues one slot down) and 1 for raw
    rows (prev[1]'s C+ terminated at the wall and became the wall
    point; the remaining C+ lines shift).  ``axis_idx`` and ``shift``
    may be traced (the raw/special decision is data-dependent).

    Unit processes are the NASA dθ-form ports: ``interior_point_nasa``
    needs prev's node ABOVE the C+ source too (the on-axis coefficient
    substitution).
    """
    W = prev.x.shape[0]
    row = jax.tree.map(jnp.zeros_like, prev)
    row = _set_node(row, 0, wall_pt)

    def body(i, row):
        ii = jnp.minimum(i + shift, W - 1)
        p1 = _row_point(prev, ii)
        p1_off = _row_point(prev, jnp.maximum(ii - 1, 0))
        p2 = _row_point(row, i - 1)
        p3 = interior_point_nasa(p1, p1_off, p2, gamma, n_corr=n_corr)
        valid = i < axis_idx
        keep = _row_point(row, i)
        sel = {k: jnp.where(valid, p3[k], keep[k]) for k in p3}
        return _set_node(row, i, sel)

    row = lax.fori_loop(1, W - 1, body, row)

    pax = axis_point_nasa(_row_point(row, axis_idx - 1), gamma,
                          n_corr=n_corr)
    row = _set_node(row, axis_idx, pax)

    fill = jnp.arange(W) > axis_idx
    return KernelRow(
        x=jnp.where(fill, row.x[axis_idx], row.x),
        r=jnp.where(fill, row.r[axis_idx], row.r),
        theta=jnp.where(fill, row.theta[axis_idx], row.theta),
        nu=jnp.where(fill, row.nu[axis_idx], row.nu),
        mu=jnp.where(fill, row.mu[axis_idx], row.mu),
        M=jnp.where(fill, row.M[axis_idx], row.M),
    )


def march_kernel(
    start_line: KernelRow,
    theta_B,
    Rt,
    Rd,
    gamma,
    n_rows: int,
    n_corr: int = 40,
    dtheta_limit: float = 0.5 * 3.141592653589793 / 180.0,
) -> KernelMarch:
    """March the throat-arc kernel with NASA's exact row policy in
    static padded arrays (see the module docstring).

    Per row, mirroring ``nasa_moc._calc_arc_wall_point_with_special``:
    compute the RAW wall point (the C+ from prev[1] terminated on the
    arc); if its step exceeds ``dtheta_limit`` (or overshoots
    ``theta_B``) take instead the SPECIAL point at
    ``min(theta_B, θ_prev + dtheta_limit/2)`` and grow the row by one;
    raw rows keep the width and shift the C+ pairing by one.  The
    raw/special decision, valid width, and pairing offset are traced;
    shapes stay static at ``W = n0 + n_rows``.  Marching freezes once
    the wall reaches ``theta_B`` (the clamped special α lands B there
    exactly); the scan carry at the end IS the BD row.

    ``n_rows`` is a static upper bound on the row count — pick
    ``ceil(theta_B/(dtheta_limit/2)) + margin`` (raw rows advance
    faster, so this always suffices; frozen rows no-op).

    θ_B-differentiability: the pre-clamp grid (raw landings and fixed
    dθ-limit steps) is θ_B-independent, so ``d(BD)/d(theta_B)`` flows
    ONLY through the final clamped row — exact and smooth within a
    row-count window; the window boundary (θ_B crossing a grid step)
    is handled by per-rung re-assembly exactly like ``kernel_bd``
    re-seeding today.
    """
    if n_rows < 1:
        raise ValueError("n_rows must be >= 1")

    n0 = start_line.x.shape[0]
    W = n0 + n_rows

    # Pad the start line with its axis state (benign sentinels).
    pad = W - n0
    sl = KernelRow(*[
        jnp.concatenate([f, jnp.full((pad,), f[-1], dtype=f.dtype)])
        for f in start_line
    ])

    def step(carry, _):
        prev, axis_idx, wall_th, done = carry

        raw = arc_wall_point_raw_nasa(prev, Rt, Rd, gamma, n_corr=n_corr)
        special = ((raw["theta"] - wall_th > dtheta_limit)
                   | (raw["theta"] > theta_B))
        alpha = jnp.minimum(theta_B, wall_th + 0.5 * dtheta_limit)
        spc = special_wall_point_nasa(prev, alpha, Rt, Rd, gamma,
                                      n_corr=n_corr)
        wall_pt = {k: jnp.where(special, spc[k], raw[k]) for k in raw}

        s = special.astype(jnp.int64)
        axis_idx_new = axis_idx + s
        row = _sweep_row_padded(wall_pt, prev, axis_idx_new, 1 - s,
                                gamma, n_corr=n_corr)

        done_row = wall_pt["theta"] >= theta_B - 1e-9
        # Freeze everything once the previous row already reached B.
        row = jax.tree.map(lambda new, old: jnp.where(done, old, new),
                           row, prev)
        axis_idx_new = jnp.where(done, axis_idx, axis_idx_new)
        wall_th_new = jnp.where(done, wall_th, wall_pt["theta"])
        carry = (row, axis_idx_new, wall_th_new, done | done_row)
        return carry, (row, axis_idx_new, wall_th_new)

    init = (sl, jnp.int64(n0 - 1), jnp.float64(start_line.theta[0]),
            jnp.bool_(False))
    (bd, bd_axis_idx, _, reached), (marched, axis_trace, _) = lax.scan(
        step, init, None, length=n_rows)

    rows = jax.tree.map(
        lambda s, m: jnp.concatenate([s[None, :], m], axis=0),
        sl, marched,
    )
    return KernelMarch(
        rows=rows, bd=bd,
        bd_axis_idx=bd_axis_idx,
        reached_wall=reached,
        row_axis_idx=jnp.concatenate(
            [jnp.array([n0 - 1]), axis_trace]),
        wall_x=rows.x[:, 0], wall_r=rows.r[:, 0],
        wall_theta=rows.theta[:, 0],
    )


def build_start_line(*args, **kwargs):
    raise NotImplementedError(
        "J3b-2: differentiable Kliegel–Levine start line (needed for "
        "d/d(Rt, gamma) design totals; theta_B sensitivity is already "
        "exact with a frozen start line) — see JAX_DIFFERENTIABLE_PLAN.md"
    )
