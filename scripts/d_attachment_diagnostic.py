"""D-attachment diagnostic for the 3e-3 stationarity floor (J4).

1. Dumps nodes 0..4 of the converged DE: x, r, M, theta, nu, mu, C+/-,
   chord slope vs characteristic slope, stationarity contribution.
2. Variants on the start condition:
     A: current theta_D pin (BD interpolation)            [baseline]
     B: drop the theta pin entirely (r pin only)
     C: C+ invariant pin  (theta0+nu0) - (theta_D+nu_D) = 0
Run: PYTHONPATH=. python scripts/d_attachment_diagnostic.py
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

import raosim.rao_variational as rv
from raosim.rao_variational import RaoSolverConfig
from raosim.jax import assembly
from raosim.jax.api import SEED_NUDGE, _logit
from raosim.gas_dynamics import prandtl_meyer, mstar_from_M
import jax
import jax.numpy as jnp
import optimistix as optx

rv.PHYSICS_WEIGHT = 1.0
G = 1.4
N = 16


def build():
    cfg = RaoSolverConfig(
        Rt=0.020, epsilon=10.0, gamma=1.4, pa_over_p0=0.01, length_pct=80.0,
        n_control=N, n_kernel=24, max_nfev=2500, residual_tol=2e-3,
        evaluate_moc=False, couple_wall=False, kernel_d_fraction_max=0.7,
        thetaN_guess_deg=24.0, formulation="characteristic")
    ce0, kbd, topo, _k = rv._initial_ce_from_kernel(cfg)
    sc = replace(cfg, kernel_bd=tuple(kbd))
    ce0.log_C = rv._seed_log_C_from_ce(ce0, cfg.gamma)
    u0 = rv._pack_bvp(ce0, -0.5, 0.01, ce0.log_C)
    from raosim.gas_dynamics import mach_from_area_ratio
    Me = mach_from_area_ratio(10.0, 1.4, supersonic=True)
    L = rv._target_length(0.02, 10.0, 80.0)
    Re = math.sqrt(10.0) * 0.02
    lower = np.concatenate([np.full(N, 1.001), np.full(N, math.radians(-10.0)),
                            np.full(N, 0.0), np.array([-1e3, -1e3, -10.0, 0.0])])
    upper = np.concatenate([np.full(N, max(12.0, 1.5 * Me)),
                            np.full(N, math.radians(65.0)), np.full(N, 1.05 * Re),
                            np.array([1e3, 1e3, 10.0, 0.7])])
    return sc, u0, lower, upper


def solve(fn_residual, u0, lower, upper, ladder=(1.0, 10.0, 30.0), n_res=None):
    lo, hi = jnp.asarray(lower), jnp.asarray(upper)
    span = hi - lo
    z = _logit(jnp.clip((jnp.asarray(u0) - lo) / span, SEED_NUDGE, 1 - SEED_NUDGE))
    solver = optx.LevenbergMarquardt(rtol=1e-10, atol=1e-12)
    for W in ladder:
        wv = np.ones(n_res)
        wv[CIDS] = W
        wj = jnp.asarray(wv)

        def fn(zz, args, _w=wj):
            return _w * fn_residual(lo + span * jax.nn.sigmoid(zz))

        s = optx.least_squares(fn, solver, z, args=None, max_steps=3000,
                               throw=False)
        z = s.value
    u = np.asarray(lo + span * jax.nn.sigmoid(z))
    return u, np.asarray(fn_residual(jnp.asarray(u)))


sc, u0, lower, upper = build()
sp = assembly.params_from_config(sc)
f0 = assembly.make_residual(sp)
SL = assembly.block_slices(sp)
CIDS = assembly.constraint_indices(sp)
n_res = int(np.asarray(f0(jnp.asarray(u0))).size)
TH_PIN = SL["ce_geometry"].start + 1   # [r_pin, theta_pin, ...]

# ---- A: baseline -----------------------------------------------------------
uA, rA = solve(f0, u0, lower, upper, n_res=n_res)
print(f"A (theta_D pin):   max={np.max(np.abs(rA)):.4g}  "
      f"alg={np.max(np.abs(rA[SL['algebraic_stationarity']])):.3e}")

# ---- node dump at A --------------------------------------------------------
M, TH, R = uA[:N], uA[N:2 * N], uA[2 * N:3 * N]
lC, kdf = uA[3 * N + 2], uA[3 * N + 3]
xD, rD, thD, MD = (float(v) for v in assembly.bd_point_at_fraction(sp, kdf))
X = np.asarray(assembly.integrate_x_from_left_mach(
    jnp.asarray(R), jnp.asarray(TH), jnp.asarray(M), xD))
nuD = prandtl_meyer(max(MD, 1.001), G)
muD = math.asin(1.0 / max(MD, 1.000001))
print(f"\nD(kernel): x={xD:.5f} r={rD:.5f} M={MD:.4f} th={math.degrees(thD):.3f} "
      f"C+={math.degrees(thD + nuD):.3f}  kdf={kdf:.4f} logC={lC:.5f}")
print("  i      x        r        M     th(deg)  C+(deg)  C-(deg)  "
      "chord_slope  char_slope   stat")
for i in range(5):
    Mi = max(M[i], 1.001)
    nu = prandtl_meyer(Mi, G)
    mu = math.asin(1.0 / Mi)
    a = math.asin(1.0 / Mi)
    stat = (math.log(mstar_from_M(Mi, G)) + math.log(abs(math.cos(TH[i] - a)))
            - math.log(math.cos(a)) - lC)
    if i < N - 1:
        dx, dr = X[i + 1] - X[i], R[i + 1] - R[i]
        chord = dr / dx if abs(dx) > 1e-15 else float("inf")
    else:
        chord = float("nan")
    char = math.tan(TH[i] + mu)
    print(f"  {i:2d}  {X[i]:.5f}  {R[i]:.5f}  {M[i]:.4f}  "
          f"{math.degrees(TH[i]):7.3f}  {math.degrees(TH[i] + nu):7.3f}  "
          f"{math.degrees(TH[i] - nu):7.3f}   {chord:9.5f}   {char:9.5f}  "
          f"{stat:+.2e}")

# ---- B: drop the theta pin --------------------------------------------------
mask = np.ones(n_res)
mask[TH_PIN] = 0.0
mj = jnp.asarray(mask)


def fB(u):
    return mj * f0(u)


uB, rB = solve(fB, u0, lower, upper, n_res=n_res)
print(f"\nB (no theta pin):  max={np.max(np.abs(rB)):.4g}  "
      f"alg={np.max(np.abs(rB[SL['algebraic_stationarity']])):.3e}")

# ---- C: C+ invariant pin ----------------------------------------------------
ONE_DEG = math.radians(1.0)


def fC(u):
    r = f0(u)
    Mj, THj = u[:N], u[N:2 * N]
    kdfj = u[3 * N + 3]
    _, _, thDj, MDj = assembly.bd_point_at_fraction(sp, kdfj)
    from raosim.jax.primitives import prandtl_meyer as pm
    inv = ((THj[0] + pm(jnp.maximum(Mj[0], 1.001), G))
           - (thDj + pm(jnp.maximum(MDj, 1.001), G))) / ONE_DEG
    return r.at[TH_PIN].set(inv)


uC, rC = solve(fC, u0, lower, upper, n_res=n_res)
print(f"C (C+ invariant):  max={np.max(np.abs(rC)):.4g}  "
      f"alg={np.max(np.abs(rC[SL['algebraic_stationarity']])):.3e}  "
      f"kdf={uC[3 * N + 3]:.4f}")
# stationarity profile under C
MC, THC = uC[:N], uC[N:2 * N]
lCC = uC[3 * N + 2]
prof = []
for i in range(N):
    Mi = max(MC[i], 1.001)
    a = math.asin(1.0 / Mi)
    prof.append(math.log(mstar_from_M(Mi, G))
                + math.log(abs(math.cos(THC[i] - a)))
                - math.log(math.cos(a)) - lCC)
print("C stat profile:", np.round(prof, 5))
