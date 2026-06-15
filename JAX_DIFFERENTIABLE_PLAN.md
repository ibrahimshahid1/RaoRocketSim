# RaoRocketSim — Differentiable (JAX) Rao Nozzle Tool

**Architecture & implementation plan.** This document is the build plan for
re-expressing the Rao optimum-thrust contour solver as an end-to-end
differentiable program in JAX, with two goals, in priority order:

1. **Tool first.** Fix the standing convergence blocker
   (`solve_rao_bvp` sits at `max_scaled ≈ 8`; the gate is `≤ 2e-3`) by
   replacing scipy's finite-difference Jacobian with exact autodiff
   Jacobians, and ship a robust contour generator.
2. **Novelty later, for free.** Exact gradients are the headline feature
   of the tool (sensitivity fields, manufacturing-tolerance maps) and the
   enabling machinery for the open research questions parked in §11.

This plan is subordinate to `REWRITE_PLAN.md`: it does not change the
physics that plan specifies, it changes the *numerics and software
substrate* underneath it. Every phase, residual, and reliability tier in
`REWRITE_PLAN.md` still applies. References to "§2.x / Phase N" below point
into that document unless stated otherwise.

---

## 0. Honest status of the idea (read this first)

We checked the literature before committing. Two corrections to earlier
assumptions:

- **"Differentiable MOC" is *not* clearly novel as a headline, but it is
  sparse.** Adjoint and surrogate optimization of MOC/CFD nozzles is
  established (chemically-reacting adjoint, 2012; MOC + free-form-deformation
  shape optimization, CEAS Space Journal 2023; NN surrogate of SERN flow with
  backprop, arXiv 2409.12707 — note that last one is a *CFD surrogate*, not
  MOC-in-a-network, contrary to how it is sometimes summarized). A fully
  autodifferentiable MOC that backpropagates *through the marching grid
  construction itself* and through the **Rao variational closure** appears
  largely unshipped, but we are **not** staking the project on that claim.
- **The interesting physics is old but under-mapped.** The fact that the
  smooth Rao optimum ceases to exist for short / large-radius nozzles and is
  replaced by a *discontinuous* solution was established by Guderley and
  co-workers (~1968). What is missing is a modern, open, numerically
  resolved characterization of *how* the optimum degrades at that boundary,
  and how real-gas γ(T) moves the boundary. Those are the §11 hooks.

Conclusion: build the differentiable tool because it is the right
engineering substrate and it fixes convergence. Treat novelty as an
application built on top, decided later.

---

## 1. Why JAX, why now

The current solver (`raosim/rao_variational.py`) builds its residual vector
from per-segment scalar functions —
`residual_Cplus_axisym`, `residual_Cminus_axisym`,
`residual_left_mach_geometry`, `residual_wall_tangency`,
`residual_cplus_child_position`, the algebraic Rao stationarity residual
(`mstar_from_M`, `gas_dynamics.py:168`), mass closure, and the valid-region
inequality — assembled in `_scaled_rao_bvp_residual` (line 2704) and
`_coupled_wall_residuals` (line 2183), then handed to
`scipy.optimize.least_squares` (line 2965 / 3559). scipy estimates the
Jacobian by **finite differences**: for a coupled-wall unknown vector with
CE + wall + characteristic nodes plus `log_C`, `λ_mdot`, `λ_length`, that is
hundreds of extra residual evaluations per iteration, each noisy near the
sonic and Mach-line singularities. That noise is the most likely reason
`max_scaled` stalls around 8 instead of descending to 2e-3.

Every one of those residual functions is **pure, composed scalar
arithmetic** (`sin`, `cos`, `atan2`, `sqrt`, `asin(1/M)`, `hypot`,
linear interpolation on arc length). That is the ideal case for reverse-mode
autodiff. Re-expressing them in `jax.numpy` gives:

- **Exact Jacobians** via `jax.jacfwd` / `jax.jacrev`, fed to a
  Gauss–Newton / Levenberg–Marquardt least-squares solver. Exact > FD for
  both convergence rate and robustness.
- **`jit`** compilation of the whole residual, so the inner solve is fast
  enough to sweep the (ε, length_pct, γ) chart.
- **Gradients of any scalar output** (Cf, Isp, wall-tangency RMS, mass
  residual) with respect to *every* design input and *every* node — the
  tool's value-add, at no extra derivation cost.

Solver library: **Optimistix** (`optimistix.LevenbergMarquardt`,
`optimistix.least_squares`, `optimistix.root_find`). It is the current,
maintained JAX nonlinear-solver library (Equinox-based; implicit-function-
theorem differentiation is the default for both forward- and reverse-mode).
`jaxopt` has broader scope but is being wound down into optax/optimistix, so
we standardize on Optimistix. (Refs in §12.)

---

## 2. Design principles

1. **Two layers, one boundary.** A pure-functional **differentiable core**
   (JAX) that owns everything between throat conditions and the emitted
   contour + residuals; and an **imperative shell** (existing NumPy/Python)
   that owns I/O, CEA, plotting, export, trajectory, CLI. Gradients only
   need to flow through the core.
2. **Oracle parity is non-negotiable.** The JAX core must reproduce the
   existing NumPy closed-form results to ~1e-10 before we trust any gradient.
   The current test suite (260 tests) is the regression oracle. This is also
   how we honor the project rule that equations be empirically grounded: the
   JAX `prandtl_meyer`, `mstar_from_M`, isentropic ratios, and Q± terms must
   match the already-literature-checked NumPy versions bit-for-bit in the
   constant-γ limit.
3. **Differentiate the *solution*, not the *iterations*.** For the converged
   BVP and for inner secant loops (θ_B, point D), use the implicit function
   theorem (Optimistix default), not backprop-through-unrolled-iterations.
   Cheaper, exact, and avoids accumulating autodiff state across hundreds of
   marching steps.
4. **No silent post-processing — enforced by construction.** The JAX core
   never rescales x_ce, never forces endpoints, never clamps θ_N. Those are
   the `REWRITE_PLAN.md` §2.F / Phase 1 sins; a differentiable rewrite is the
   natural moment to delete them for good, because endpoint closure becomes a
   residual the solver actually satisfies.
5. **Incremental, not big-bang.** Land the core behind a feature flag
   (`solver_backend="jax"`) next to the existing path. The NumPy path stays
   the reference until the JAX path passes the same gates.

---

## 3. The port boundary — what moves to JAX vs stays NumPy

| Module / concern | Move to JAX core? | Rationale |
|---|---|---|
| `gas_dynamics.py` (isentropic ratios, `prandtl_meyer`, `mach_from_prandtl_meyer`, `mstar_from_M`, `mach_angle`, area–Mach) | **Yes** (mirror module `gas_dynamics_jax.py`) | These are the leaf primitives every residual differentiates through. |
| `rao_residuals.py` (C±, left-Mach, wall-tangency, intersection) | **Yes** | The residual leaves; trivial, pure scalar math. |
| Rao algebraic stationarity + differential consistency (§2.B) | **Yes** | Core of the optimum condition; needs gradients. |
| Mass closure `curve_mass_flux` / kernel-BD integral (§2.D) | **Yes** | Integrand differentiable; D becomes a solved unknown. |
| Valid-region inequality (§2.E) | **Yes** (as a smooth diagnostic) | Used for gating + the §11 cliff study (needs ∂/∂design). |
| `_pack_bvp` / `_unpack_bvp` / `_coupled_wall_residuals` / `_scaled_rao_bvp_residual` | **Yes** (re-expressed as flat array ops) | The residual assembly is the function we autodiff + jit. |
| BVP solve (currently scipy `least_squares`) | **Yes** (Optimistix LM) | The whole point. |
| `transonic_kernel.py` (Kliegel–Levine start line) | **Yes** | Feeds the seed; differentiable so start-line sensitivity is available (§11). |
| `moc.py` unit processes (`solve_interior_point`, `solve_axis_point`, `solve_wall_point`) | **Port the math, keep the marching as `lax.scan`/`while_loop`** | Sequential, data-dependent length — see §5. |
| `nasa_moc.py` (NASA port, kernel/topology, `set_theta_b`, secant loops) | **Partial** — port the numerics, wrap secant loops with implicit diff | Reference + topology; gradients only needed through the converged result. |
| `cea.py`, `propellants.py`, `physics.py` (Bartz), `thermo.py` (Phase 10) | **No (call as host functions)** | CEA is an external equilibrium solver; not differentiable. Use frozen γ inside the core; consult CEA outside it. Real-gas hook in §11. |
| `legacy_io.py`, `export.py`, `plotting.py`, `trajectory.py`, `altitude_performance.py`, `trade_study.py`, CLI (`main.py`, `design.py`) | **No** | I/O and orchestration; no gradients needed. They consume the core's output dict. |
| `benchmarks.py`, test suite | **No** (but extend) | Stay as the oracle; add JAX-path parity + convergence tests. |

Net: a focused new subpackage `raosim/jax/` holding ~6 files
(`primitives.py`, `moc_kernel.py`, `residuals.py`, `bvp.py`, `pack.py`,
`api.py`). Everything else is untouched or lightly adapted.

---

## 4. Differentiable core architecture

```
raosim/jax/
    primitives.py   # gas dynamics in jax.numpy, sonic-safe
    moc_kernel.py   # transonic start line + characteristic march (lax.scan)
    residuals.py    # all Rao/MOC residual blocks, vectorized over nodes
    pack.py         # flat-vector <-> structured state (pytree) packing
    bvp.py          # Optimistix LM solve + implicit-diff wrapper
    api.py          # solve_rao_bvp_jax(config) -> RaoSolution (same dict shape)
```

### 4.1 Primitives (`primitives.py`)

Reimplement, in `jax.numpy`, exactly the closed forms already in
`gas_dynamics.py`: `isentropic_{temperature,pressure,density}_ratio`,
`prandtl_meyer`, `mstar_from_M`, `mach_angle`, `area_mach_relation`. The
constant-γ Prandtl–Meyer and M* forms are the ones already cross-checked
against `prmeyer.pdf` / `rao1999.pdf`, so parity to 1e-10 with the NumPy
versions is the acceptance test (Phase J1).

**Sonic-safety.** `mach_angle = asin(1/M)` and the `sqrt(M²−1)` in
Prandtl–Meyer are singular/branch-sensitive at M→1. Use guarded forms
(`jnp.sqrt(jnp.maximum(M*M - 1, 0))`, clamp `1/M` to `(0,1]`) and a
`jax.custom_jvp` on the throat-region primitives so the derivative stays
finite at M=1. `mach_from_prandtl_meyer` (currently a Newton iterate at
`gas_dynamics.py:198`) becomes either a fixed `lax.scan` of Newton steps or
an Optimistix `root_find` with implicit diff.

### 4.2 MOC kernel & march (`moc_kernel.py`)

Port the throat start line (`transonic_kernel.py`, Kliegel–Levine, with the
documented NASA `KLThroat` typo correction noted in `REWRITE_PLAN.md` §8.1)
and the axisymmetric unit processes from `moc.py`. The Anderson Q± source
terms (`moc.py:154-160`) are already the literature ground truth and port
verbatim. The march itself — variable number of characteristic rows — is the
one genuinely sequential, data-dependent piece; handle per §5 with a fixed
`lax.scan` over a padded grid.

### 4.3 Residual stack (`residuals.py`)

One pure function:

```python
def rao_residual(state: BVPState, params: DesignParams) -> jax.Array:
    """Flat residual vector r(u; p). Blocks, in REWRITE_PLAN priority order:
      - endpoint closure        (ce[-1]=wall[-1]=(L,Re); wall[0]=(Nx,Ny))
      - Rao algebraic stationarity at each CE node      (§2.B)
      - left-Mach-line geometry on CE                   (§2.C)
      - axisymmetric C+/C- compatibility on CE & wall   (§2.A)
      - mass closure CE vs kernel BD                    (§2.D)
      - wall tangency + flow-angle match at wall        (§2.F)
      - characteristic-intersection closure             (§2.F)
      - (smoothness regularizer, tiny weight 0.02)      (§2.A)
    """
```

This is the existing `_coupled_wall_residuals` + `_scaled_rao_bvp_residual`
content, but vectorized with `vmap` over node pairs instead of Python list
comprehensions, and returning a single concatenated array. Identical scaling
(`L_scale`, `Re_scale`, `theta_scale = radians(1°)`) so residual magnitudes
match the current run and the 2e-3 gate is directly comparable.

### 4.4 BVP solve (`bvp.py`)

```python
solver = optimistix.LevenbergMarquardt(rtol=1e-8, atol=1e-10)
sol = optimistix.least_squares(
    lambda u, p: rao_residual(unpack(u, p), p),
    solver, y0=u_seed, args=params, max_steps=...,
)
```

Exact Jacobian comes from Optimistix's Gauss–Newton/LM step using
`jax.jacfwd` internally. Implicit-function-theorem differentiation of the
converged `u*(p)` w.r.t. `params` is automatic — that is what powers the
gradient API in §6.

### 4.5 Inner iterations via implicit diff

`set_theta_b` (outer secant on θ_B) and the point-D mass-matching secant
(`calc_lrc_de` / `find_point_e`) are root-finds. Wrap each as
`optimistix.root_find`; their gradients then come from the implicit function
theorem, so we never backprop through the secant iterations. This is the
single most important trick for keeping the autodiff graph small and the
derivatives exact.

---

## 5. Autodiff hazards & mitigations

| Hazard | Where | Mitigation |
|---|---|---|
| **Data-dependent loop length** (number of characteristic rows depends on geometry) | MOC march, kernel build | Fix a maximum grid size; `lax.scan` over a padded array with a validity mask. Masked nodes contribute zero residual. JAX requires static shapes under `jit`. |
| **Non-smooth post-processing** (monotonic-x filter, endpoint forcing, θ_N clamp) | `_construct_wall_from_ce`, `resample_wall_for_export`, `to_contour_dict` | **Delete them** in the JAX path (they are `REWRITE_PLAN.md` defects anyway). Endpoint closure becomes a residual, not a fixup. |
| **Sonic singularities** (`asin(1/M)`, `sqrt(M²−1)`) | primitives near throat | Guarded ops + `custom_jvp` finite-derivative overrides (§4.1). |
| **`atan2`, `hypot` at coincident nodes** (ds→0) | residual segments | `eps`-floors already present in NumPy (`max(..., 1e-9)`, `1e-12`); keep them, they are differentiable. |
| **`min`/`max` in valid-region inequality** (non-smooth at the active segment) | §2.E gate | For *gating*, keep hard `min`. For the §11 *sensitivity study*, use a smooth-min (log-sum-exp) variant so ∂(boundary)/∂p is defined. |
| **Branch selection** (`mach_from_area_ratio` sub/supersonic branch) | wall seed | Select branch outside the differentiated region (in the seed builder), pass the chosen branch as a static flag. |
| **NaN poisoning** (one bad node NaNs the whole Jacobian) | everywhere | `jax.debug` checks in tests; `jnp.where` guards; unit parity tests per primitive before integration. |

---

## 6. The gradient API (the tool's headline feature)

Once `u*(params)` is differentiable, expose:

```python
# raosim/jax/api.py
sensitivities = rao_sensitivities(config)   # one call, all gradients

sensitivities.dCf_dparams      # ∂Cf/∂(ε, length_pct, Rt, γ, θ_B, ...)
sensitivities.dCf_dwall        # ∂Cf/∂r at every wall node  -> tolerance field
sensitivities.dIsp_dgamma      # real-gas sensitivity (scalar)
sensitivities.jacobian         # full ∂r/∂u (for diagnostics / conditioning)
sensitivities.hessian_thrust   # ∂²(thrust)/∂(wall)²  (for §11 soft-mode study)
```

Immediate tool features that fall out:

- **Manufacturing-tolerance map.** Paint `|∂Cf/∂r_node|` onto the contour:
  which millimeters of the bell actually move performance. Add
  `plot_sensitivity_field(solution)` to `plotting.py` (Phase 13 already wants
  flowfield plots; this slots in).
- **Gradient-based design.** `maximize Cf s.t. (length, ε)` becomes a
  handful of Optimistix steps with exact gradients instead of the current
  scipy SLSQP wall optimizer (`rao_optimizer.py`, slated for deprecation in
  `REWRITE_PLAN.md` Phase 8).
- **Exact ∂/∂γ.** Sets up the Phase 10 real-gas work cleanly.

---

## 7. Validation strategy

Gates, in order — each must pass before the next:

1. **J1 — primitive parity.** `gas_dynamics_jax` matches `gas_dynamics` to
   1e-10 across M∈[1.01, 6], γ∈{1.14…1.4}. (`tests/test_jax_primitives.py`)
2. **J2 — residual parity.** `rao_residual` on a fixed state equals the
   NumPy `_scaled_rao_bvp_residual` + `_coupled_wall_residuals` to 1e-8.
   This proves the port changed *nothing physical*. (`tests/test_jax_residual_parity.py`)
3. **J3 — Jacobian sanity.** `jax.jacfwd(rao_residual)` matches a
   central-difference Jacobian to 1e-5 on a few states. Confirms autodiff is
   wired right. (`tests/test_jax_jacobian.py`)
4. **J4 — convergence (the payoff).** On the ε=10 / length_pct=80 / γ=1.4
   reference case, Optimistix LM drives `max_scaled` **below 2e-3** with
   `moc_compatibility_preserved=True` and `postprocessed=False`. This is the
   `REWRITE_PLAN.md` Phase 6 gate that currently xfails
   (`test_phase6_coupled_wall.py:378`). (`tests/test_jax_convergence.py`)
5. **J5 — chart benchmark.** Re-run Phase 7: θ_N/θ_E across the
   (ε, length_pct) grid within RMS 1.5° / max 3°
   (`test_rao_chart_benchmark_plan_targets`, currently xfail). Must use
   `angle_boundary_mode="free"` (REWRITE_PLAN §13.1) so it isn't circular.
6. **J6 — NASA oracle (later).** Feed the JAX path through
   `scripts/compare_nasa_reference.py` against `outputs_M3.5Perf/` once the
   kernel port matches (Phase 12 territory).

The whole existing 260-test suite must stay green throughout — the JAX path
is additive until it earns the default.

---

## 8. Phasing (maps onto REWRITE_PLAN phases)

| JAX phase | Deliverable | REWRITE_PLAN tie-in | Gate |
|---|---|---|---|
| **J0** | `raosim/jax/` skeleton, deps, `solver_backend` flag, CI job | infra | imports + smoke |
| **J1** | `primitives.py` + parity tests | substrate for all | J1 |
| **J2** | `residuals.py` + `pack.py`, residual parity | §2.A–F re-expressed | J2 |
| **J3** | `bvp.py` Optimistix LM, exact Jacobian, implicit diff | Phase 6 numerics | J3 |
| **J4** | **Convergence on reference case** | **closes Phase 6 xfail** | **J4** |
| **J5** | chart sweep under JAX | closes Phase 7 plan-target xfail | J5 |
| **J6** | gradient API + `plot_sensitivity_field` | Phase 13 + tool value | smoke + a known-sign sensitivity |
| **J7** | (optional) real-gas ∂/∂γ, NASA oracle | Phase 10 / 12 | J6 / parity |

J0–J4 is the spike that proves the thesis (exact Jacobians fix
convergence). J5–J6 turn it into the tool. J7 is the on-ramp to the §11
research.

---

## 9. Dependencies & environment

Add to `requirements.txt`: `jax`, `jaxlib`, `optimistix`, `equinox`.
(`jax[cpu]` is sufficient; no GPU needed at this problem size.)

Note: the checked-in `.venv` references Python 3.14 and is not portable —
stand up a fresh environment (`python -m venv .venv && pip install -r
requirements.txt`) for the JAX work. Pin JAX/jaxlib versions in
`requirements.txt` since their API moves; record the exact versions in the
J0 commit.

---

## 10. Risks & mitigations

- **Risk: exact Jacobians converge but to a *different* (non-Rao) point.**
  Mitigation: J2 residual parity guarantees identical physics; J5 chart
  benchmark guarantees the converged point is the published Rao optimum.
- **Risk: the marching grid's static-shape requirement bloats memory or
  hurts accuracy via padding.** Mitigation: size the padded grid from the
  known `lastKernelJ ≈ 57` of the M3.5Perf reference; mask is exact, padding
  only costs compute.
- **Risk: scope creep into a full JAX rewrite.** Mitigation: the §3 boundary
  is deliberately narrow; CEA/export/plotting/trajectory stay NumPy forever.
- **Risk: convergence still stalls (Jacobian wasn't the problem).** Then the
  blocker is the *seed* / kernel TT′ mismatch (`docs/legacy_code_audit.md`:
  Python axis-point M≈1.27 vs NASA 1.5), not the optimizer. J4 failing would
  *diagnose* this cleanly because exact Jacobians remove the FD-noise
  confound — itself a useful result.

---

## 11. Preserved novelty hooks (build later, on top of the tool)

The differentiable core makes these cheap; none are required for the tool.

1. **Soft-mode at the Guderley cliff.** Use `hessian_thrust` to track the
   spectrum of ∂²(thrust)/∂(contour)² as (ε, length_pct) approaches the
   valid-region boundary (§2.E). Hypothesis: a near-zero eigenvalue with a
   characterizable spatial shape appears as the smooth optimum loses
   existence — a variational bifurcation the 1968 theory predicts but never
   computed at this resolution. *Honest framing:* the boundary is classically
   known (Guderley); the **sensitivity/stability structure across it** is the
   open part.
2. **Real-gas migration of the cliff.** With ∂/∂γ available and the Phase 10
   `thermo.py` in place, map how the existence boundary itself moves between
   frozen and shifting-equilibrium chemistry for kerolox/methalox. The
   classical boundary is perfect-gas only.
3. **Start-line provenance propagation.** Exact ∂Cf/∂(start-line model
   params) quantifies how much Sauer vs Hall vs Kliegel–Levine actually
   changes the final optimum — a long-hand-waved modeling choice, now
   measurable.

---

## 12. Open decisions for you

1. **Where does `raosim/jax/` live** — new subpackage (recommended) vs
   methods inside existing modules?
2. **Default switch policy** — keep NumPy default until J5 passes (safe), or
   flip to JAX once J4 passes (faster iteration, more risk)?
3. **Padded-grid size** — fix from the M3.5Perf reference, or make it a
   config field?
4. **Pursue any §11 hook now**, or strictly tool-first until J6 ships?

---

### References

Repo: `REWRITE_PLAN.md` (governing physics plan, §2.A–F, Phases 6–7, §8.1,
§13.1); `docs/legacy_code_audit.md` (TT′ seed blocker); `raosim/gas_dynamics.py`
(`mstar_from_M:168`, `prandtl_meyer:152`); `raosim/moc.py:154-160` (Anderson Q±);
`raosim/rao_variational.py` (`_coupled_wall_residuals:2183`,
`_scaled_rao_bvp_residual:2704`, scipy `least_squares:2965`).

Literature (`propulsion_texts/`): Rao, *Exhaust Nozzle Contour for Optimum
Thrust*, Jet Propulsion 1958 (`RaoRecentDevinRockNozConfig.pdf` lineage);
Rao/Beck/Booth AIAA 99-2584 (`rao1999.pdf`); Östlund, KTH 2002
(`fulltext01.pdf`); Rice, JHU/APL RTDC-TPS-481 2003 (`20030067852.pdf`);
*Contoured Rocket Nozzles* (`978-3-7091-4745-0_18.pdf`).

External (verified June 2026): Guderley et al., *Continuous and discontinuous
solutions for optimum thrust nozzles of given length*, JOTA —
https://link.springer.com/article/10.1007/BF00934781 ; Optimistix (Rader,
Lyons, Kidger), arXiv 2402.09983, docs https://docs.kidger.site/optimistix/ ;
MOC + free-form-deformation shape optimization, CEAS Space Journal 2023 —
https://link.springer.com/article/10.1007/s12567-023-00511-1 ; adjoint design
in reacting flow —
https://www.sciencedirect.com/science/article/abs/pii/S0045793012001880 .

*Document owner: (you). Update phase gates as J0–J7 land. Keep §0's honesty
note — it is why the project is scoped tool-first.*
