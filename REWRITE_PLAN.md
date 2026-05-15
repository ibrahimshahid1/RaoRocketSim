# RaoRocketSim — Rao Variational / Contour Pipeline Rewrite Plan

This plan addresses the real mathematical and engineering defects in
`raosim/rao_variational.py` and the contour-generation pipeline that consumes
it (`raosim/moc.py`, `raosim/rao_optimizer.py`, `raosim/nozzle_geometry.py`,
`raosim/validation.py`, the test suite in `tests/`, and the entry points in
`main.py` and `design.py`).

It is structured so each phase can be landed independently. Each phase ends
with a test gate. Do not advance reliability until that gate passes.

---

## 0. What the other model got wrong

Before fixing anything, you should know that **most of the "this function
doesn't exist, add it" claims in the other model's reply are wrong**. They
already exist in this repo. The model was operating without the rest of the
codebase, so it inferred a much smaller, much rougher baseline.

| Symbol the model said to "add new" | Actual status in repo |
|---|---|
| `solve_rao_bvp` | Already exists — `raosim/rao_variational.py:1219` |
| `RaoBVPSolution` (proposed) | Equivalent already exists as `RaoSolution` — `rao_variational.py:163` |
| `FlowNode` (proposed dataclass) | Equivalent already exists as `CharPoint` — `raosim/moc.py:55` |
| `prandtl_meyer` | Already in `raosim/gas_dynamics.py:152` |
| `mach_angle` | Already in `gas_dynamics.py:168` |
| `mach_from_prandtl_meyer` | Already in `gas_dynamics.py:175` |
| `count_characteristic_crossings` | Already exists as `check_characteristic_crossing` — `rao_variational.py:1201` |
| `_compatibility_smoothness` (model wanted to rename) | Already exists at `rao_variational.py:1031` — the rename is the right fix, see §2.A |
| `optimize_wall` | Already exists — `raosim/rao_optimizer.py:114` (uses scipy SLSQP) |
| `moc_bell_nozzle` | Already exists — `rao_optimizer.py:286` |
| `_full_cone_length`, `_lookup_theta_n` | Already exist — `rao_optimizer.py:39, 44` (and `rao_variational.py:1445`) |
| `rao_variational_moc_contour` | Already exists — `rao_variational.py:1450` |
| Axisymmetric C+/C− compatibility with Q± source terms | Already in `raosim/moc.py:105-241` (`solve_interior_point`, `solve_axis_point`, `solve_wall_point`) using the correct Anderson-style Q± = ±sin(θ)·sin(μ)·cos(μ)/(r·cos(θ±μ)). |
| Kernel-seeded CE initial guess | Already exists — `_initial_ce_from_kernel` at `rao_variational.py:958` |
| `least_squares`-based BVP solve | Already wired — `rao_variational.py:1274` |
| Lagrange multipliers for mass + length | Already in `ControlSurface` (`lambda2`, `lambda3`) and used in `solve_optimal_control_surface` and `_scaled_rao_bvp_residual`. |
| Wall tangency RMS metric | Already in `_wall_tangency_rms` at `rao_variational.py:1173` |
| Stationarity / transversality residuals | Already in `stationarity_residuals` (line 384) and `transversality_residual` (line 418), using Euler-Lagrange of (f₁ + λ₂f₂ + λ₃f₃). |
| `RaoInvalidRegionError` exception | Not present — but the reliability enum already encodes invalid-region semantics via `ContourReliability.GEOMETRIC_APPROXIMATION`. A dedicated exception is optional, not required. |
| `optimize_wall_legacy` (proposed deprecation shim) | Not needed; `optimize_wall` itself should be marked legacy and its public callers (`moc_bell_nozzle`, `bell_nozzle_contour(method='moc')`) should be re-routed or annotated. |

So the rewrite is not "add 20 missing functions." It is **fix the math that
the existing functions encode and add the four Rao closure equations that
are genuinely missing**.

---

## 1. Scope of damage — what actually generates an untrustworthy contour

End-to-end, the contour-generation pipeline for `method='rao_variational_moc'`
is:

```
bell_nozzle_contour(method='rao_variational_moc')        # nozzle_geometry.py:185
  → rao_variational_moc_contour                          # rao_variational.py:1450
  → solve_rao_bvp                                        # rao_variational.py:1219
       _initial_ce_from_kernel                           # rao_variational.py:958
       least_squares(_scaled_rao_bvp_residual, …)        # rao_variational.py:1274 + 1042
            _integrate_ce → thrust/massflow/length       # rao_variational.py:475 + 279/316/336
            _stationarity_matrix                         # rao_variational.py:1018 (Euler-Lagrange via finite differences)
            transversality_residual                      # rao_variational.py:418
            _compatibility_smoothness                    # rao_variational.py:1031   ← MISLABELED, see §2.A
            (incidence + monotonic-M + φ-smoothness penalties)
       construct_wall_from_ce_raw → _construct_wall_from_ce  # rao_variational.py:688
            approximate_starting_line (area_ratio default)    # moc.py
            interior + axis + wall march, then CE-driven wall # uses correct Q± in moc.py
            x_ce LINEARLY RESCALED to span [Nx+0.05L, L]      # rao_variational.py:745-749
            monotonic-x filter; endpoint enforced; monotonic-r cleanup
       resample_wall_for_export
            ENDPOINTS FORCED again to (Nx,Ny), (L,Re)         # rao_variational.py:1162-1169
  → solution.to_contour_dict
  → add_contour_reliability_metadata
```

In short: even when the BVP residual fails to converge, the wall comes out
because every downstream stage performs a geometric cleanup that hides the
failure. The reliability flag drops to `GEOMETRIC_APPROXIMATION`, but
nothing prevents the caller from exporting STL/STEP and treating the result
as manufacturable.

That is the heart of the problem: **convergence is decoupled from contour
emission**. Hardware-grade output requires (i) no silent post-processing
and (ii) the math underneath the residual must actually be Rao's optimum-
thrust system, not a regularized smoothness penalty.

---

## 2. Real mathematical defects (with line numbers)

### 2.A `_compatibility_smoothness` is regularization, not compatibility — `rao_variational.py:1031`

Current:

```python
def _compatibility_smoothness(ce, gamma):
    nu = np.array([prandtl_meyer(...) for M in ce.M])
    kp = ce.theta + nu
    km = ce.theta - nu
    return np.concatenate([np.diff(kp, n=2), np.diff(km, n=2)]) / scale
```

This is a 2nd-difference penalty on θ ± ν along CE. In **planar** flow, θ ± ν
are exact Riemann invariants along C±, so a 2nd-difference penalty pushes
them toward linear — which is closer to compatibility but isn't compatibility.
In **axisymmetric** flow (which is what this code targets), θ ± ν are not
even local invariants; the source terms

```
Q⁺ = +sin(θ)·sin(μ)·cos(μ) / (r·cos(θ+μ))
Q⁻ = −sin(θ)·sin(μ)·cos(μ) / (r·cos(θ−μ))
```

(as used correctly in `moc.py:154-160`) make the right-hand sides
spatially varying. So the penalty does not enforce any physical
compatibility on the CE — it only enforces smoothness.

**Fix.** Two parts:

1. Rename: `_compatibility_smoothness` → `_ce_smoothness_regularization`,
   and drop its claim of "MOC compatibility" in the docstring. Lower its
   weight (it should be a tiny stabilizer, not a primary residual).
2. Replace its role in the residual stack with the **axisymmetric MOC
   compatibility along each CE segment**, using the same Q± formulas
   already in `moc.py`. Two new residual functions:

```python
def residual_Cminus_axisym(p0, p1, gamma):
    # along C−: d(θ − ν) = Q⁻·ds
    nu0 = prandtl_meyer(p0.M, gamma);  nu1 = prandtl_meyer(p1.M, gamma)
    lhs = (p1.theta - nu1) - (p0.theta - nu0)
    ds = math.hypot(p1.x - p0.x, p1.r - p0.r)
    th = 0.5*(p0.theta + p1.theta); mu = 0.5*(p0.mu + p1.mu); r = max(0.5*(p0.r + p1.r), 1e-9)
    cos_tm = math.cos(th - mu)
    Qm = -math.sin(th)*math.sin(mu)*math.cos(mu) / (r * cos_tm) if abs(cos_tm) > 1e-12 else 0.0
    return lhs - Qm * ds

def residual_Cplus_axisym(p0, p1, gamma):
    # along C+: d(θ + ν) = Q+·ds
    ...
```

These are not new physics — they are exactly what `solve_interior_point` /
`solve_wall_point` already do in `moc.py`. We are just promoting them from
"local construction step" to "global residual on CE / wall segments."

### 2.B Algebraic Rao optimum-thrust stationarity is missing

The current `stationarity_residuals` (line 384) computes the calculus-of-
variations Euler-Lagrange equations numerically (central differences of
the integrands). That is mathematically equivalent to the Rao condition in
the continuous limit but is numerically fragile because `_numerical_partials`
uses dh = 1e-6 on integrand values that include the singular `1/sin(φ)` term
in `length_integrand`, and the φ-derivative residual is dominated by
`−λ₃ / sin²(φ)` which the test suite explicitly excludes (`test_rao_variational.py:88-91`).

The Rao/Beck/Booth AIAA 99-2584 form (your `rao1999.pdf`) is closed-form
and much better-conditioned. For a perfect gas with `M* = V/a* = √[(γ+1)M² / (2 + (γ−1)M²)]`:

```
M* · cos(θ − α) / cos(α) = C            (algebraic stationarity along DE)
```

Add this as an additional residual block in `_scaled_rao_bvp_residual`,
with `log(C)` as a new unknown in the solve vector (one scalar). The
differential form,

```
d(ln M*) − (dθ − dα)·tan(θ−α) + dα·tan(α) = 0
```

is the segment-to-segment consistency check; keep it as a secondary residual
during development, then retire it once the algebraic form alone is robust.

Status: **net new**. Not present anywhere in the repo.

### 2.C DE is not enforced to be a left-running Mach line

In Rao's optimum-thrust formulation, the supersonic control surface DE is
exactly a left-running Mach line. Geometrically:

```
dr/dx |_{DE} = tan(θ + α)        where α = μ = arcsin(1/M)
```

Currently `ce.phi` is a free unknown in the BVP, and the only constraints
on it are heuristic: `phi ≥ theta + 0.25°` (incidence penalty, line 1066),
2nd-difference smoothness (line 1069), bounds (line 1247-1253). Nothing
forces φ to equal `θ + α`.

Add a hard residual along CE:

```python
def residual_left_mach_geometry(p0, p1):
    dx = p1.x - p0.x;  dr = p1.r - p0.r
    th = 0.5*(p0.theta + p1.theta);  mu = 0.5*(p0.mu + p1.mu)
    return dr - dx * math.tan(th + mu)
```

This couples φ to (θ, M) properly. The current `phi` becomes auxiliary:
either eliminate it from the unknowns entirely (compute `phi = arctan(dr/dx)` from
adjacent CE nodes), or keep it but add `residual_phi_equals_theta_plus_mu`.
Eliminating it is cleaner and reduces the unknown count by `n_control`.

Status: **net new**. The `phi_theta_gap` penalty on line 1066 is a
one-sided incidence guard, not the Mach-line constraint.

### 2.D Mass closure uses the wrong target

`_target_mdot` (line 951) computes `ρV* · A_t` (quasi-1D throat mass flow).
But the Rao condition is that **the mass flux integrated over the control
surface DE equals the mass flux through the kernel cross-section BD** —
two physical curves in the same flowfield. The throat A_t value is only
an asymptote and ignores the kernel's axisymmetric correction.

Replace with a discrete integral over both surfaces:

```python
def curve_mass_flux(nodes, gamma):
    # axisymmetric: dṁ = 2π R ρV sin(β − θ) ds
    # along a left Mach line: β = θ + α, so sin(β − θ) = sin(α).
    total = 0.0
    for p0, p1 in zip(nodes[:-1], nodes[1:]):
        ds = math.hypot(p1.x - p0.x, p1.r - p0.r)
        if ds < 1e-12: continue
        beta = math.atan2(p1.r - p0.r, p1.x - p0.x)
        M = 0.5*(p0.M + p1.M); theta = 0.5*(p0.theta + p1.theta)
        r = max(0.5*(p0.r + p1.r), 1e-9)
        rho = isentropic_density_ratio(M, gamma)
        T = isentropic_temperature_ratio(M, gamma)
        V = M * math.sqrt(gamma * T)        # non-dimensional
        total += 2.0*math.pi * r * rho * V * abs(math.sin(beta - theta)) * ds
    return total
```

And the residual:

```python
mdot_residual = (curve_mass_flux(ce_nodes, gamma)
                 - curve_mass_flux(kernel_BD_segment, gamma)) / m_ref
```

The kernel BD segment is the subset of `_initial_ce_from_kernel`'s
`kernel_points` whose radius spans D (CE start) → B (axis intersection on
the kernel last left-Mach-line). That subset selection is the new "choose D"
step — D is no longer a heuristic 5%-past-N point like the current
`x_ce[0] = Nx + 0.05*(Ln - Nx)` on line 734.

Status: **partly new**. `massflow_integrand` exists (line 316) but is only
ever integrated radially over CE (line 502), not as a surface-normal flux,
and the target is the throat A_t value. Both the integrand variant (with
`sin(β − θ)`) and the kernel-side integration are new.

### 2.E Rao valid-region inequality is never evaluated

The Rao optimum exists only when

```
1 − (dα/dθ) · [tan(θ−α) + tan(α)] / [tan(θ−α) − tan(α)]   ≥ 0
```

holds along DE. For very short / over-expanded nozzles the optimum
contour is discontinuous and the smooth-flow Rao construction is
inapplicable. Currently `solve_rao_bvp` will happily return a
`RAO_VARIATIONAL_RESIDUAL_SOLVED` reliability for inputs in the
invalid region — there is no check anywhere in the file.

Add a post-solve evaluator:

```python
def rao_valid_region(ce, tol=0.0):
    bvalues = []
    for p0, p1 in zip(ce[:-1], ce[1:]):
        dth = p1.theta - p0.theta
        if abs(dth) < 1e-10:
            continue
        a0 = math.asin(1.0/p0.M); a1 = math.asin(1.0/p1.M)
        th = 0.5*(p0.theta + p1.theta); a = 0.5*(a0 + a1)
        num = math.tan(th - a) + math.tan(a)
        den = math.tan(th - a) - math.tan(a)
        if abs(den) < 1e-12:
            bvalues.append(-math.inf)
        else:
            bvalues.append(1.0 - (a1 - a0)/dth * (num/den))
    return min(bvalues) if bvalues else float("inf"), bvalues
```

Use it in `solve_rao_bvp` to gate reliability:

```python
boundary_min, _ = rao_valid_region(...)
if boundary_min < -tol:
    reliability = ContourReliability.GEOMETRIC_APPROXIMATION
    warnings.append("Requested (ε, length_pct) is outside the smooth-flow Rao region.")
```

Status: **net new**.

### 2.F Wall is constructed sequentially, not coupled, and is silently post-processed

`_construct_wall_from_ce` (lines 688-939) does the right MOC unit processes
locally (it uses `moc.solve_interior_point` and `moc.solve_axis_point`,
which carry correct Q± terms). But:

1. It linearly **rescales x_ce after the fact** to span `[Nx+0.05L, L]`
   (lines 744-749). This is a coordinate change that breaks the physical
   meaning of φ as the local CE tangent direction. After this scaling, the
   CE no longer satisfies the constraint that produced it.
2. It drops non-monotonic-x wall points (line 896-902) and flips
   `moc_compatibility_preserved = False`. Fine as a diagnostic, but the
   downstream gate at line 1377-1390 still allows reliability to drop to
   `MOC_COMPATIBLE` even when wall tangency RMS is large.
3. `resample_wall_for_export` (line 1141) forces endpoints to `(Nx, Ny)`
   and `(L, Re)` regardless of what the raw wall did (lines 1162-1169).
   Sets a diagnostic flag but emits the contour anyway.
4. `to_contour_dict` (line 182) appends conv + throat arcs **using the
   theta_n that the solver found** (line 198), but then forces
   `theta_n = max(theta_N, 15°)`. If the solver wanted a smaller throat-
   downstream tangent, that is silently overridden. The conv + throat
   arcs are re-built every call, so if the solver disagrees with the
   prepended arcs at point N, the seam at N is geometrically continuous
   but not slope-continuous. This is exactly the type of seam that fails
   manufacturing inspection on a real bell.

**Fix.** Three things, in order:

1. **Remove `x_ce` rescaling on lines 744-749.** If x_ce doesn't land at
   `(Ln, Re)` naturally, that is a residual failure and should propagate.
2. **Promote the wall to a joint unknown in the BVP** (this is the model's
   "coupled" suggestion, applied correctly). The cleanest formulation:
   the BVP unknown vector becomes
   ```
   u = [M_ce, theta_ce, x_ce, r_ce,
        x_wall, r_wall, M_wall, theta_wall,
        x_char, r_char, M_char, theta_char,
        log_C_rao, lambda_mdot, lambda_length]
   ```
   with residuals (in priority order):
   - Endpoint closure: ce[-1] = wall[-1] = (Ln, Re)
   - Rao algebraic stationarity at each CE node (§2.B)
   - Left-Mach-line geometry on CE (§2.C)
   - Axisymmetric MOC compatibility C+/C− on CE and wall (§2.A)
   - Mass closure between CE and kernel BD (§2.D)
   - Characteristic-intersection closure inside the kernel triangle
   - Wall tangency: dr_wall/dx_wall = tan(θ_wall) and θ_wall = θ_flow at wall
   - Transversality at E (re-derived for fixed (L, Re), §2.I)
   - φ = θ + α (or eliminate φ entirely)
3. **Stop forcing endpoints in export.** `resample_wall_for_export`
   should refuse to emit a contour whose raw endpoints differ from
   targets by more than residual_tol. Replace the silent override with
   a `RaoEndpointMismatchError`.

Status: changes to existing code (lines 745-749, 919-920, 930, 1162-1169);
plus a substantial restructure of `_pack_bvp` / `_unpack_bvp` to carry
wall and characteristic nodes.

### 2.G `to_contour_dict` rebuilds conv + throat arcs using a clamped theta_n

Line 198: `theta_n = max(self.theta_N, math.radians(15.0))`. If the
solver wanted θ_N = 12°, the conv+throat arcs are still drawn at 15°,
producing a slope discontinuity at N. The bell starts at the solver's
θ_N but the upstream arc ends at 15° — a kink.

Fix: either let θ_N float below 15° (drop the floor) and propagate it
to the upstream arc construction, or constrain the solver to
θ_N ≥ 15° explicitly. The current behavior is the worst of both worlds.

### 2.H Starting-line default is the weakest option

`solve_rao_bvp` defaults to `starting_line_method='area_ratio'`
(`RaoSolverConfig`, line 132). The docstring of `moc.py` (lines 30-38)
itself notes that `'hall'` is the more physically correct option for
curved-throat transonic flow. The default should be `'hall'` once the
Hall correction is benchmarked, or a new `'kliegel_levine'` path should
be added (best-in-class transonic startline; see Östlund §3 — fulltext01.pdf).

### 2.I Transversality formulation is off for fixed-length, fixed-ε

`transversality_residual` (line 418) is `f₁ + λ₂f₂ + λ₃f₃` at the last
CE node. This is the natural transversality condition for a **free**
endpoint. But the current solver fixes both `L_target` and `Re`, so E is
fully specified: `x_E = L`, `r_E = Re`. With both endpoints fixed, the
natural transversality condition is automatically satisfied by the
endpoint closure conditions; the residual on line 418 is double-counting.

Fix: drop the transversality residual from the residual stack when the
solver is in fixed-length / fixed-ε mode (the only mode currently used).
Keep the function for a future variable-length mode where it is
mathematically correct.

### 2.J `rao_variational_contour` (the legacy path) should be deprecated

The legacy `rao_variational_contour` (line 1504) does **not** use the
BVP. It uses `solve_optimal_control_surface` (line 509), a loose gradient-
descent on Lagrange multipliers, plus the `_initial_ce_guess` (heuristic
θ from 20° linear; line 437) rather than the kernel-seeded init. It is
exposed via `bell_nozzle_contour(method='rao')` (`nozzle_geometry.py:175`)
and is mathematically much weaker than `solve_rao_bvp`.

Mark it deprecated, emit a DeprecationWarning, and route
`bell_nozzle_contour(method='rao')` to `rao_variational_moc_contour`
once the new BVP path is benchmarked.

### 2.K Test suite doesn't actually exercise the BVP

`tests/test_rao_variational_moc.py` calls `solve_rao_bvp` with
`max_nfev=0` and `evaluate_moc=False`. That returns the initial residual
without running the least-squares solver and without constructing the MOC
wall. The tests pass without any actual physics. Add convergence tests
with `max_nfev ≥ 200` and `evaluate_moc=True`, asserting that
`residuals.max_scaled <= residual_tol` and that
`construction_diagnostics["moc_compatibility_preserved"]` is True.

### 2.L Benchmarks against Rao 1958 / NASA SP-8120 chart not wired in

`raosim/benchmarks.py` exists (707 lines) but is not in the BVP test
loop. After §2.A–§2.F land, add a calibration script that:

- Picks (Rt, ε, length_pct) from the (ε, length_pct) grid in
  `nozzle_geometry.py:31-46` (which IS the NASA SP-8120 chart).
- Runs `solve_rao_bvp` on each grid point.
- Compares the solver's θ_N and θ_E against the chart (`_THETA_N_TABLE`,
  `_THETA_E_TABLE`).
- Fails (xfail → fail) if RMS error > 2°. The chart is the published
  Rao reference; the solver must reproduce it before claiming
  `RAO_VARIATIONAL_RESIDUAL_SOLVED`.

---

## 3. Literature grounding

Files present in `/Users/ibrahimshahid/Downloads/propulsion_texts`:

| File | What to use it for |
|---|---|
| `rao1999.pdf` (≈ AIAA 99-2584, Rao/Beck/Booth) | Closed-form Rao stationarity (§2.B), valid-region inequality (§2.E), differential stationarity, mass-closure statement. The single most important PDF for this rewrite. |
| `RaoRecentDevinRockNozConfig.pdf` (Rao, ARS J. 1961) | Original optimum-thrust derivation. Cross-check the algebraic stationarity sign conventions. |
| NASA SP reports (`19710019929.pdf`, `19830016278.pdf`, `19900015790.pdf`, `19970004933.pdf`, `20030067852.pdf`) | At least one is SP-8120 — look for the θ_n/θ_e chart that matches `_THETA_N_TABLE`. Use as benchmark target. |
| `fulltext01.pdf` (likely Östlund, KTH 2002) | NPE shock-free region, kernel/control-surface terminology, Hall/Kliegel-Levine starting-line discussion. Use for §2.E and §2.H. |
| `prmeyer.pdf` | Cross-check `prandtl_meyer` sign convention. |
| `nozzle_geometries.pdf`, `Statement+of+the+problem+...pdf` | Background on dual-bell and variational nozzle problems. |
| `L-05_BSplines_NURBS.pdf`, `CAD_04.pdf` | Spline interpolation for contour export (manufacturable contour resampling). |

**Anderson's *Modern Compressible Flow* is NOT in your folder**, despite
the other model citing it. The repo's own `moc.py:18-23` already encodes
the Anderson axisymmetric Q± formulas correctly — those are the
ground truth for §2.A.

---

## 4. Phased implementation plan

Each phase is independently landable. Do not advance to the next phase
until the current phase's test gate passes.

### Phase 1 — Honest naming + reliability gates (no math change)

Goal: stop the silent failure modes before changing any equations.

1. Rename `_compatibility_smoothness` → `_ce_smoothness_regularization`
   (`rao_variational.py:1031`). Update docstring to say "this is a
   regularization, not MOC compatibility." Update its weight from 1.0
   to 0.02 in the residual stack (line 1079). Update the diagnostic
   field name `compatibility_rms` → `regularization_rms` in
   `RaoResidualReport` (line 137) and propagate.
2. Add `RaoInvalidRegionError(RuntimeError)` (optional but useful).
3. In `solve_rao_bvp`:
   - Refuse to emit `RAO_VARIATIONAL_RESIDUAL_SOLVED` unless
     `construction_diagnostics["postprocessed"]` is False and
     `construction_diagnostics["moc_compatibility_preserved"]` is True
     (currently only the second is checked — line 1378).
   - Refuse to emit `MOC_COMPATIBLE` unless
     `construction_diagnostics["postprocessed"]` is False.
4. In `resample_wall_for_export` (line 1141):
   - Compute the raw endpoint deltas before forcing them.
   - If `|r_export[0] - start[1]| > residual_tol * Re` or
     `|r_export[-1] - end[1]| > residual_tol * Re`, raise
     `RaoEndpointMismatchError`. Do NOT silently force endpoints.
5. In `to_contour_dict` (line 182):
   - Drop the `max(self.theta_N, 15°)` floor on line 198. Use the
     solver's θ_N as-is. If θ_N < 15° is physically nonsense for the
     chosen (ε, length_pct), the BVP residual should be the thing
     that rejects it.
6. Update `test_rao_variational_moc.py` to use `max_nfev >= 200` and
   `evaluate_moc=True`. Mark the current "initial residual only" tests
   as `@pytest.mark.smoke`.

**Test gate (Phase 1):** existing tests still pass with new gates;
no contour with `RAO_VARIATIONAL_RESIDUAL_SOLVED` claim escapes
without a post-processed=False diagnostic.

### Phase 2 — Axisymmetric MOC compatibility on CE and wall

Goal: residual stack uses physical compatibility, not just smoothness.

1. Add a `FlowNode` *adapter dataclass* on top of `CharPoint` (do not
   replace `CharPoint`). `FlowNode` carries `x, r, M, theta` plus a
   computed `mu`, and is what the new residual functions consume.
   Add `to_flow_node()` on `CharPoint`.
2. Add to `rao_variational.py` (or a new submodule
   `raosim/rao_residuals.py`):
   - `residual_Cplus_axisym(p0, p1, gamma) -> float`
   - `residual_Cminus_axisym(p0, p1, gamma) -> float`
   - `residual_left_mach_geometry(p0, p1) -> float`     (§2.C)
   - `residual_wall_tangency(w0, w1) -> float`          (§2.F)
   - `residual_intersection(p_plus, p_minus, child, x_scale, r_scale)` (§2.F)
3. Add unit tests in a new `tests/test_rao_residuals.py`:
   - **Planar limit**: drop the source term, build (p0, p1) on a single
     C+ or C− line in planar (Q = 0) flow — residual must be < 1e-10.
   - **Axisymmetric self-consistency**: take the `solve_interior_point`
     output for known parents; verify `residual_Cplus_axisym(parent_plus,
     child, γ)` and `residual_Cminus_axisym(parent_minus, child, γ)`
     are both < 1e-6 (i.e., the residuals agree with the local march).
4. Wire `residual_Cplus_axisym` / `residual_Cminus_axisym` into
   `_scaled_rao_bvp_residual` as new residual blocks. Reduce the weight
   of `_ce_smoothness_regularization` to 0.02 (it becomes a stabilizer).

**Test gate (Phase 2):** new tests pass. Existing BVP test
(`test_rao_bvp_solution_is_auditable_and_not_hardware_qualified`) with
`max_nfev = 200`, `evaluate_moc = True` produces
`residuals.max_scaled < 5e-3` for (Rt=0.020, ε=10, length_pct=80, γ=1.4).

### Phase 3 — Algebraic Rao stationarity + left-Mach-line geometry

Goal: solver uses the Rao closed-form optimum condition, not the numerically
fragile Euler-Lagrange partials.

1. In `gas_dynamics.py`, add `mstar_from_M(M, gamma)`.
2. Add to `rao_variational.py`:
   - `rao_stationarity_residual(node, log_C, gamma)`              (§2.B, algebraic)
   - `rao_stationarity_fd_residual(p0, p1, gamma)`                (§2.B, differential)
   - `rao_control_surface_compatibility_residual(p0, p1, gamma)`  (§2.B Eq. 4)
3. Promote `log_C_rao` to a solver unknown in `_pack_bvp` /
   `_unpack_bvp`.
4. Replace the numerical Euler-Lagrange `stationarity_residuals` in the
   residual stack with the algebraic Rao form. Keep `stationarity_residuals`
   in the file as a reference implementation, marked `# reference: not used in residual`.
5. Add `residual_left_mach_geometry` along CE. Either:
   - Keep `phi` as an unknown and add `residual_phi_equals_theta_plus_mu`, or
   - Eliminate `phi` from the unknown vector (compute it from adjacent
     (x, r) — preferred for fewer DOFs).
6. Drop `transversality_residual` from the residual stack for fixed-length
   fixed-ε runs (§2.I). Keep endpoint closure conditions.

**Test gate (Phase 3):** for (ε=10, length_pct=80, γ=1.4), the converged
θ_N from the BVP must match `_THETA_N_TABLE` lookup (which gives 30°) within ±2°.
Likewise θ_E vs `_THETA_E_TABLE` (15.5°) within ±2°.

### Phase 4 — Mass closure on the kernel cross-section

Goal: replace the quasi-1D throat target with the actual surface integral.

1. Add `curve_mass_flux(nodes, gamma)` per §2.D.
2. In `_initial_ce_from_kernel`, return the kernel **left-Mach-line BD**
   not just the starting line. (The current `kernel_points` is the
   starting line; BD is the last left-Mach-line in the kernel, which
   is what mass must match through.)
3. Replace the `_target_mdot` residual at line 1074 with
   `(curve_mass_flux(ce) - curve_mass_flux(kernel_BD)) / mdot_ref`.
4. The point D itself becomes a solved-for unknown (its position on the
   kernel BD line is whatever makes mass match).

**Test gate (Phase 4):** mass residual < 1e-4 for the (ε=10, length_pct=80)
case. Solver still matches NASA chart within ±2°. Thrust coefficient
from CE surface integration matches `thrust_coefficient()` from
`gas_dynamics.py` within 0.5%.

### Phase 5 — Valid-region check + reliability cliff

Goal: invalid Rao inputs (very short / very over-expanded) fail
explicitly rather than producing a smoothed pseudo-contour.

1. Add `rao_valid_region(ce_nodes, tol=0)` per §2.E.
2. In `solve_rao_bvp`, after the residual report, evaluate the boundary.
   If `min_boundary < -residual_tol`, emit:
   - `reliability = ContourReliability.GEOMETRIC_APPROXIMATION`
   - `rao_region = "invalid_short_nozzle_region"`
   - `requires_discontinuous_exit_flow_model = True`
   - explicit warning string.
3. Add a new test that picks an (ε, length_pct) in the discontinuous
   region (e.g. ε=4, length_pct=60) and asserts `rao_region` is invalid.

**Test gate (Phase 5):** valid-region check fires on at least one chart
corner and does not fire on the (ε=10, length_pct=80) reference case.

### Phase 6 — Coupled wall in the BVP (no more sequential construction)

Goal: drop the silent post-processing path entirely.

1. Extend the unknown vector `u` to carry `(x_wall, r_wall, M_wall, theta_wall)`
   and a sparse characteristic net `(x_ch, r_ch, M_ch, theta_ch)` at one
   child node per CE/wall strip.
2. Add the wall and characteristic-intersection residuals from §2.F.
3. Remove the linear `x_ce` rescaling from `_construct_wall_from_ce`
   (lines 744-749). After Phase 6 the entire `_construct_wall_from_ce`
   collapses into a thin post-processor that just extracts the wall
   subarray from the solved `u`.
4. Phase out `resample_wall_for_export`'s endpoint forcing. The new
   behavior: if endpoint mismatch exceeds residual_tol, raise; otherwise
   the wall is already on the targets and no resample is needed.

**Test gate (Phase 6):** `construction_diagnostics["postprocessed"]` is
False for all converged solutions of the (ε ∈ {5, 10, 15}, length_pct ∈
{70, 80, 90}) sweep. Wall tangency RMS < 0.1° everywhere.

### Phase 7 — Benchmark validation + reliability promotion

Goal: earn the right to call any of this `BENCHMARK_VALIDATED`.

1. In `raosim/benchmarks.py`, add `rao_variational_chart_benchmark()`:
   sweep the (ε, length_pct) grid from `_THETA_N_TABLE` / `_THETA_E_TABLE`,
   run `solve_rao_bvp` on each, compare θ_N and θ_E.
2. Add `tests/test_rao_chart_benchmark.py` that runs the full sweep
   (mark `@pytest.mark.slow`). RMS error in θ_N AND θ_E must be < 1.5°,
   max error < 3°.
3. Once that test passes for a release, the reliability gate inside
   `solve_rao_bvp` may emit `BENCHMARK_VALIDATED` when the per-run
   residuals are < 1e-4 and the input is on (or interpolable within) the
   benchmarked grid.

### Phase 8 — Deprecate the weak paths

1. `rao_variational_contour` (line 1504) → `DeprecationWarning`, route
   `bell_nozzle_contour(method='rao')` (`nozzle_geometry.py:175`) to
   `rao_variational_moc_contour`.
2. `rao_optimizer.optimize_wall` (line 114) → `DeprecationWarning`.
   `bell_nozzle_contour(method='moc')` continues to work but emits the
   warning at import time. The Bezier+optimized-angles path
   (`moc_bell_nozzle`) is fine for preliminary geometry; flag that
   explicitly in the contour dict's `design_status`.
3. Remove `solve_optimal_control_surface` (line 509) once Phase 6 is
   stable — the BVP supersedes it.

---

## 5. Test plan summary

| Phase | New tests | Location |
|---|---|---|
| 1 | `test_endpoint_mismatch_raises`, `test_postprocessed_does_not_claim_residual_solved` | `tests/test_rao_variational_moc.py` |
| 2 | `test_residual_Cplus_planar_invariant`, `test_residual_Cminus_planar_invariant`, `test_residual_Cplus_axisym_matches_interior_march` | `tests/test_rao_residuals.py` (new) |
| 3 | `test_algebraic_stationarity_at_converged_optimum`, `test_left_mach_geometry_holds_on_CE`, `test_chart_corner_theta_n_e10_l80` | `tests/test_rao_variational_moc.py` |
| 4 | `test_mass_closure_residual_small`, `test_thrust_coeff_matches_quasi1d` | `tests/test_rao_variational_moc.py` |
| 5 | `test_valid_region_fires_on_short_nozzle`, `test_valid_region_quiet_on_e10_l80` | `tests/test_rao_variational_moc.py` |
| 6 | `test_no_postprocessing_on_chart_sweep` | `tests/test_rao_variational_moc.py` |
| 7 | `test_rao_chart_benchmark_full_grid` | `tests/test_rao_chart_benchmark.py` (new, `@slow`) |
| 8 | `test_deprecation_warning_rao_variational_contour`, `test_deprecation_warning_optimize_wall` | `tests/test_deprecations.py` (new) |

Run `pytest -m "not slow"` in CI; the chart benchmark runs nightly /
on release.

---

## 6. API and contour-emission contract

After Phase 6, the contour dict emitted by `rao_variational_moc_contour`
gains:

- `rao_region`: `"valid_shock_free_region"` | `"invalid_short_nozzle_region"`
- `requires_discontinuous_exit_flow_model`: bool
- `boundary_min`: float (Rao valid-region inequality, min over CE)
- `mass_closure_residual_rel`: float
- `stationarity_constant_C`: float (the converged Rao optimum constant)
- `wall_tangency_rms_deg`: float (already exists as `wall_tangency_rms`, just convert units)

The export pipeline (`raosim/export.py`) gets a small upgrade: it should
refuse to emit STL / STEP for `reliability < RAO_VARIATIONAL_RESIDUAL_SOLVED`
unless an explicit `allow_preliminary_export=True` flag is set. This
makes "manufacturable contour" a property of the API contract, not a
hope.

---

## 7. Order of work — what to do first

If you only have a weekend, do Phase 1 (honest naming + reliability gates +
no-silent-postprocess) and Phase 2 (axisymmetric compatibility residuals).
Those two together neutralize the most dangerous failure mode (silent
emission of a non-Rao contour) without requiring the algebraic Rao math
to land yet.

If you have a week, add Phase 3 (algebraic stationarity + left-Mach-line).
That is where the contour actually starts to match published Rao results.

If you have a month, do everything through Phase 7 and ship the chart
benchmark. At that point — and only at that point — the
`rao_variational_moc` path is publishable as a Rao-method contour
generator. Until then, the contour is a "Rao-flavored variational guess"
and should be labeled as such (which the current `design_status =
"experimental_rao_variational_moc_bvp"` already does, but most users
will not read).

---

## 8. Phase 9 — Kliegel-Levine transonic kernel

The current `'hall'` starting-line in `raosim/moc.py:322-400` (the
`approximate_starting_line` function) is **mislabeled**. What it
actually implements is a leading-order Hall first-term plus a
quadratic correction:

```python
# moc.py:371-385  (paraphrased)
rho_c = Rd / Rt
a1 = math.sqrt(2.0 / (gp1 * rho_c))      # Hall first-order coefficient
a2 = gp1 / (12.0 * rho_c)                # second-order, simplified
xi = (r - Rt) / Rt
M = 1.0 + a1 * xi + a2 * xi * xi
```

This is closer to **Sauer (1947)** than Hall, and it is *not* Kliegel-
Levine. The docstring already concedes "this implementation is
intentionally simplified" (`moc.py:36`). For tight-throat nozzles
(`Rd/Rt = 0.382` in this codebase — that is the entire problem domain)
the leading-order Hall form has a known accuracy cliff below Rc/Rt ≈ 2.

### 8.1 What to implement

Replace the body of `approximate_starting_line(..., method='hall')`
with a full Kliegel-Levine 1969 expansion through third order. The
KL series is in inverse powers of `(R + 1)` where `R = Rc/Rt` (Hall
used `1/R`; the KL reformulation extends convergence down to small R
and removes a sign error in Hall's third-order term):

```
u/a* = 1 +  A1(r,z;γ) / (R+1)   +  A2(r,z;γ) / (R+1)²  +  A3(r,z;γ) / (R+1)³
v/a* =      B1(r,z;γ) / (R+1)   +  B2(r,z;γ) / (R+1)²  +  B3(r,z;γ) / (R+1)³
```

The explicit polynomial coefficients `A1..A3, B1..B3` are tabulated in:

- **Zucrow & Hoffman, *Gas Dynamics Vol. 2*, Ch. 16** — coefficient-by-
  coefficient through third order. Primary source.
- **Östlund, KTH thesis 2002** (this is your `fulltext01.pdf`) —
  reproduces the Kliegel-Levine coefficients. **Use as primary source
  because it is in the repo's literature folder.**
- **Kliegel & Levine, *AIAA Journal* 7(7) 1375-1378, 1969**, DOI
  10.2514/3.5355 — the original paper. Open-access PDF:
  <https://arc.aiaa.org/doi/pdf/10.2514/3.5355>
- **Kliegel NTRS report AED-R-71-10** —
  <https://ntrs.nasa.gov/api/citations/19710024785/downloads/19710024785.pdf>
- **Sivells AEDC nozzle design code documentation** (DTIC ADA062944) —
  the FORTRAN parent of most modern MOC nozzle codes; contains an
  explicit Hall/KL kernel.

A Python reference port worth diffing against:
<https://github.com/noahess/conturpy> (port of Sivells' Contur).

### 8.2 Implementation plan

1. New file `raosim/transonic_kernel.py`:

   ```python
   from dataclasses import dataclass

   @dataclass(frozen=True)
   class KliegelLevineState:
       u_over_astar: float   # axial velocity / sonic speed at throat
       v_over_astar: float   # radial velocity / sonic speed
       M: float
       theta: float          # = atan(v/u)
       r: float
       x: float
       gamma: float
       R_throat_curvature: float

   def kliegel_levine_state(
       r_over_rt: float,
       z_over_rt: float,
       gamma: float,
       R_curvature_ratio: float,  # Rc/Rt = Rd/Rt
       order: int = 3,
   ) -> KliegelLevineState:
       """
       Kliegel-Levine (1969) transonic expansion at (r, z) near the throat.
       Reference: Östlund 2002 §3.x (Zucrow & Hoffman Ch. 16 coefficients).
       """
       ...
   ```

   Implement `A1..A3` and `B1..B3` exactly as tabulated in Östlund
   §3 (page numbers to be filled in by the implementer after reading
   `fulltext01.pdf`). Use `math.fma` or careful Horner evaluation for
   the polynomial sums — the coefficients are ill-conditioned for
   small R+1.

2. Add `method='kliegel_levine'` to `approximate_starting_line` in
   `raosim/moc.py:322`. Implementation:

   ```python
   if method == 'kliegel_levine':
       Rc = Rd
       for ang in angles:
           x = Rd * math.cos(ang - math.pi/2)
           r = (Rt + Rd) + Rd * math.sin(ang - math.pi/2)
           kl = kliegel_levine_state(
               r_over_rt = r / Rt,
               z_over_rt = x / Rt,
               gamma = gamma,
               R_curvature_ratio = Rc / Rt,
               order = 3,
           )
           pt = _make_point(x, r, kl.theta, kl.M, gamma)
           points.append(pt)
   ```

3. Rename the existing `method='hall'` branch to `method='sauer_hall1'`
   (truth in labeling — it is the leading-order Hall reduction, which
   is the Sauer-class solution). Keep it for backwards compatibility
   but emit a deprecation warning when selected.

4. Promote `method='kliegel_levine'` to the default in
   `RaoSolverConfig.starting_line_method` (`rao_variational.py:132`)
   and in `bell_nozzle_contour` (`nozzle_geometry.py:115`). Update the
   docstrings on `moc.py:30-38` to reflect the new default and accuracy
   envelope.

### 8.3 Validation

1. **Unit tests** in `tests/test_transonic_kernel.py`:
   - At `r = 0` (axis), `v = 0` exactly. `u/a*` at axis is the standard
     centerline transonic velocity; check against the
     Zucrow-Hoffman tabulated value for `Rc/Rt = 2.0, γ = 1.4`.
   - In the limit `Rc/Rt → ∞`, KL must reduce to the 1-D sonic
     line (`u/a* = 1, v = 0` everywhere on the throat plane).
   - The KL Mach distribution along the throat arc must be monotonic
     in `θ` from axis to wall.

2. **Cross-check against Cuffel-Back-Massier (1969) measurements**.
   The CBM data is in
   <https://drahmednagib.com/onewebmedia/SPC407/CUFFEL_1969.pdf>.
   For `Rc/Rt = 0.625` (their tightest case), the centerline Mach
   immediately past the throat must agree within ±2%.

3. **End-to-end test**: with KL as the starting line, the BVP
   convergence behavior on the (ε=10, length_pct=80) reference case
   should not degrade. Compare residual norm and wall tangency RMS
   before and after the switch.

### 8.4 Test gate (Phase 9)

KL starting line is default. `Sauer_hall1` still works but warns.
Unit tests pass. End-to-end residuals are no worse than the
`'area_ratio'` baseline for the chart sweep.

---

## 9. Phase 10 — Real-gas / variable-γ MOC (frozen + shifting equilibrium hooks)

The current MOC and Rao residual stack assumes calorically perfect
ideal gas with a single `gamma` argument plumbed through every call
site. Real combustion-product flow has γ(T) varying from ~1.14 at
chamber stagnation to ~1.30 at the exit for typical kerolox / methalox.
The 15-30% γ shift changes θ_N, θ_E, c_star, and especially the
Bartz heat flux significantly.

The repo already has `raosim/cea.py` (RocketCEA wrapper with frozen
and shifting equilibrium modes), `raosim/propellants.py` (built-in
table), and `raosim/physics.py` (Bartz screening). The Phase 10 work
is to **extend `gamma` from a scalar argument to a `ThermoModel`
object** that is consulted at each MOC node.

### 9.1 What to implement

New file `raosim/thermo.py`:

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Callable

class ThermoModel(Protocol):
    """Calorically perfect or NASA-9-polynomial frozen thermochemistry."""
    def gamma(self, T: float) -> float: ...
    def cp(self, T: float) -> float: ...
    def molecular_weight(self) -> float: ...
    def R_specific(self) -> float: ...

    # Convenience: T(M) along the local isentrope through stagnation T0
    def temperature_from_mach(self, M: float, T0: float) -> float: ...
    # Numerically integrated Prandtl-Meyer for variable-γ.
    def prandtl_meyer(self, M: float, T0: float) -> float: ...
    def mach_from_prandtl_meyer(self, nu: float, T0: float) -> float: ...


@dataclass(frozen=True)
class ConstantGammaThermo:
    gamma_const: float
    Mw: float
    def gamma(self, T): return self.gamma_const
    ...  # closed-form Prandtl-Meyer (existing gas_dynamics.prandtl_meyer)


@dataclass(frozen=True)
class FrozenCEAThermo:
    """
    Composition frozen at chamber, γ(T)/Cp(T) from NASA 9-coeff polys.
    Reference: Gordon & McBride, NASA RP-1311 Part I, 1994.
    """
    a_low: tuple[float, ...]    # NASA-9 coefficients, low-T range
    a_high: tuple[float, ...]
    T_switch: float
    Mw: float
    T_ref: float = 298.15
    def cp(self, T): ...
    def gamma(self, T): return self.cp(T) / (self.cp(T) - self.R_specific())
    def prandtl_meyer(self, M, T0):
        # numerically integrate dν = sqrt(M²-1)/(1 + (γ(T)-1)/2 · M²) · dM/M
        # along the isentrope T(M) = T0 / (1 + (γ-1)/2 M²)  with γ updated each step
        ...
```

Wire `ThermoModel` into:

- `raosim/gas_dynamics.py` — overload-by-keyword: if `gamma` is a
  `ThermoModel` instance, dispatch to the model's `prandtl_meyer`;
  otherwise keep the closed-form path.
- `raosim/moc.py` — every `solve_*_point` accepts a `ThermoModel`.
  `_make_point` uses `thermo.prandtl_meyer(M, T0)` instead of the
  scalar-γ form. The compatibility-equation form is unchanged (it is
  Riemann-invariant in θ ± ν regardless of γ structure — see Zebbiche
  & Salhi, *Proc. IMechE Part G*, 2016, doi:10.1177/0954410016636913).
- `raosim/rao_variational.py` — the Rao stationarity residual uses
  `M*(M; γ_local)`. For the algebraic Rao condition, `M*` is the local
  dimensionless velocity, so:

  ```python
  def mstar_local(M, thermo, T0):
      gamma_local = thermo.gamma(thermo.temperature_from_mach(M, T0))
      return math.sqrt((gamma_local + 1) * M*M / (2 + (gamma_local - 1) * M*M))
  ```

- `raosim/physics.py` — `bartz_heat_flux` should use **shifting-
  equilibrium c_p** at the local static T, not chamber c_p. This is a
  10-30% effect. Pass through `ShiftingEquilibriumThermo`, which
  internally calls CEA at the local (T, p) point.

### 9.2 Three thermo modes

1. **`constant_gamma`** — current behavior, no change. Tests still pass.
2. **`frozen_cea`** — composition fixed at chamber (RocketCEA's frozen
   flag = 1 in `get_Throat_MolWt_gamma`). γ(T) varies via NASA-9 polys.
   This is the default for hardware-grade design work.
3. **`shifting_equilibrium`** — calls CEA at each local (p, T) point.
   Slow. Recommended only for the post-converged Bartz/c_star check,
   not inside the MOC inner loop. The MOC march itself runs frozen;
   shifting-equilibrium is only used for performance prediction and
   heat-transfer validation.

For finite-rate chemistry, scope a **Bray sudden-freezing** wrapper
(see Bray, *JFM* 6, 1959; Sarli/Burwell/Zupnik AIAA 1965-554) that
runs equilibrium until a freezing Mach `M_freeze` and frozen afterward.
Defer to a follow-on phase if needed.

### 9.3 Reference data sources

- **NASA RP-1311 Part I (Analysis)** — Gordon & McBride 1994, NASA TM-
  4513 source of the 9-coefficient polynomials.
  <https://www.grc.nasa.gov/WWW/CEAWeb/RP-1311.pdf>
  <https://shepherd.caltech.edu/EDL/PublicResources/sdt/refs/NASA-TM-4513.pdf>
- **NASA RP-1311 Part II (User Manual)** — CEA usage, including the
  `frozen` and `equilibrium` modes used in this codebase's `cea.py`.
- **RocketCEA API**:
  <https://rocketcea.readthedocs.io/en/latest/functions.html>
  Confirmed available calls:
  - `get_Chamber_MolWt_gamma(Pc, MR)` (already used at `cea.py:77`)
  - `get_Throat_MolWt_gamma(Pc, MR, frozen=0|1)` — add this call
  - `get_Throat_Transport(Pc, MR, frozen=0|1)` — needed for Bartz with
    shifting equilibrium
  - `get_eq_PambEval(Pc, MR, eps, Pamb)` — for design-condition exit
    state with ambient back pressure
- **Zebbiche & Salhi**, *Proc. IMechE Part G* 230(13) 2016 — generalized
  Prandtl-Meyer with variable γ.
  <https://journals.sagepub.com/doi/10.1177/0954410016636913>
- **MIT 16.512 lecture 14** — frozen vs shifting in nozzle calcs.
  <https://ocw.mit.edu/courses/16-512-rocket-propulsion-fall-2005/466f2a7b69434d563dc78f39b9be9bb8_lecture_14.pdf>
- **NASA TR R-132 (Svehla 1962)** — transport properties for high-T
  combustion products.
- **NASA TN D-2599 (Bartz 1965)** for Bartz correction with variable
  transport.

### 9.4 Test plan

`tests/test_thermo.py` (new):
1. `test_constant_gamma_matches_existing_prandtl_meyer` — same numerical
   value as the closed-form `gas_dynamics.prandtl_meyer` to 1e-12.
2. `test_frozen_cea_thermo_recovers_constant_gamma_in_limit` — when
   NASA-9 coefficients are set to a calorically perfect gas, the
   numerically integrated PM function matches the closed-form.
3. `test_frozen_cea_gamma_decreases_with_T_for_combustion_products`.
4. `test_shifting_equilibrium_thrust_coefficient_matches_cea_directly`
   — CEA's published `IvacCstr` for LOX/CH4 at MR=3.5, Pc=10MPa, ε=40
   within 0.2%.
5. `test_bartz_shift_equilibrium_higher_than_chamber_frozen` — assert
   shifting-equilibrium Bartz heat flux is 10-30% higher than chamber-
   frozen-γ Bartz at the throat.

### 9.5 Test gate (Phase 10)

All three thermo modes selectable via `RaoSolverConfig.thermo_mode`.
Constant-γ tests still pass. CEA-frozen mode reproduces NASA RP-1104
(your `19830016278.pdf`) chart θ_N within ±2° at γ_effective = 1.20
(common kerolox value), demonstrating the chart is γ-dependent as
expected. Shifting-equilibrium Isp matches CEA's `get_IvacCstr` within
0.5%.

---

## 10. Phase 11 — Supersonic CFD cross-check (SU2 axisymmetric Euler / RANS)

CFD is the gate that promotes a Rao contour from
`BENCHMARK_VALIDATED` (matches NASA chart) to `CFD_CHECKED` (matches a
high-fidelity flow solver). This is *not* a "write a CFD solver"
workstream — it is "define the interface, the mesh, the BCs, and the
tolerances, then call an existing solver."

### 10.1 Solver choice: SU2

After surveying the open-source options (SU2 vs OpenFOAM `rhoCentralFoam`
vs older AGARD codes), **SU2 is the right pick**, for three reasons:

1. **Native axisymmetric switch.** `AXISYMMETRIC= YES` in the .cfg
   adds the geometric source terms automatically. `rhoCentralFoam`
   requires a wedge mesh (a 5° pie slice), which is more error-prone
   for nozzle validation.
2. **Density-based, multiple flux schemes** (Roe, AUSM+, JST). Matches
   what the MOC compatibility equations encode.
3. **Python wrapper (`pysu2`)** is supported and lets you launch a
   run, drive it iteratively, and extract wall pressure / Mach from
   memory without a VTU round-trip.

References:
- SU2 theory: <https://su2code.github.io/docs_v7/Theory/>
- SU2 axisymmetric: <https://github.com/su2code/SU2/issues/324>
- SU2 supersonic outflow (Giles non-reflecting): <https://github.com/su2code/SU2/issues/717>
- pySU2 wrapper: <https://su2code.github.io/docs/Python-Wrapper-Build/>
- SU2 SWBLI V&V (same family of solver settings):
  <https://su2code.github.io/vandv/swbli/>

### 10.2 Benchmark reference

**NASA RP-1104** (Hoffman 1983, *Perfect Bell Nozzle Parametric and
Optimization Curves*) — and this is `/Users/ibrahimshahid/Downloads/
propulsion_texts/19830016278.pdf` **already in your literature folder**.
RP-1104 publishes MOC reference contours and `Cf` for the (ε, length_pct)
chart. That is the CFD comparison target.

Secondary: Aspirespace TOP nozzle MOC reference (γ=1.4),
<http://www.aspirespace.org.uk/downloads/Thrust%20optimised%20parabolic%20nozzle.pdf>.

### 10.3 What to implement

New module `raosim/cfd/` (new subpackage):

```
raosim/cfd/
    __init__.py
    mesh.py          # axisymmetric quad mesh generator (Gmsh .geo writer)
    su2_runner.py    # writes .cfg, launches pysu2, reads results
    compare.py       # CFD-vs-MOC tolerance gates
    cases/
        e10_l80_g14.json   # canonical validation case definitions
        ...
```

- `mesh.py` writes a Gmsh `.geo` file: nozzle wall = contour from the
  Rao solver, axis at `r=0`, inlet far-field box upstream of the
  throat, outlet plane some Re downstream of the exit. Output `.su2`
  via `gmsh -2 -format su2`.
- `su2_runner.py` writes a SU2 configuration with:
  - `SOLVER= EULER` (inviscid Euler for MOC matching) or
    `SOLVER= RANS, KIND_TURB_MODEL= SST` for the Bartz check.
  - `AXISYMMETRIC= YES`
  - `MARKER_EULER= ( wall )` (slip wall)
  - `MARKER_SYM= ( axis )` (axis symmetry)
  - `MARKER_INLET= ( inlet, T0, P0, ... )` with `TOTAL_CONDITIONS_PT`
  - `MARKER_SUPERSONIC_OUTLET= ( outlet, ... )` for the supersonic
    exit, or Giles non-reflecting for subsonic transients.
  - `CONV_FIELD= RMS_DENSITY`
  - Mesh sizing per §10.4 below.
- `compare.py` ingests the SU2 wall-pressure and exit-plane Mach
  arrays and computes:
  - `Cf_cfd` vs `Cf_moc` (target: within ±0.5%)
  - Wall pressure RMS vs MOC streamline pressure (target: within ±3%)
  - Exit Mach RMS (target: within ±1%)

### 10.4 Mesh resolution

For axisymmetric Euler on a Rao nozzle at ε=10:
- 200-300 cells from the throat to the exit, axially.
- 80-120 cells radial.
- Geometric clustering toward the throat (growth ratio 1.1).
- ~100k structured quads total. Grid-independent.

For RANS (Bartz validation):
- First cell wall-normal spacing for y+ < 1.
- 20-30 prism layers.
- Cell count ~500k.

References:
- SimuTech compressible meshing guidance:
  <https://simutechgroup.com/compressible-flow-cfd-for-aerospace/>
- Mesh y+ guidance:
  <https://www.leapaust.com.au/blog/cfd/tips-tricks-turbulence-wall-functions-and-y-requirements/>

### 10.5 Tolerances and reliability promotion

Published Rao-nozzle CFD V&V studies (EUCASS altitude-optimization
work) report **thrust numerical uncertainty < 0.10%** after grid
independence. The gate to promote a contour to `CFD_CHECKED`:

- `|Cf_cfd - Cf_moc| / Cf_moc < 0.5%`
- `wall_pressure_rms_rel < 3%` (relative to chamber pressure)
- `exit_mach_rms_rel < 1%`
- `wall_pressure_monotonic_post_throat == True` (no spurious shocks)

For RANS+Bartz validation:
- `q_wall_max_cfd / q_wall_max_bartz_screen` between 0.7 and 1.5
  (Bartz is screening; CFD is reference).

EUCASS reference: <https://www.eucass.eu/component/docindexer/?task=download&id=4222>

### 10.6 CI strategy

- CFD runs are too slow for per-commit CI. Mark as `@pytest.mark.cfd`
  and run them only:
  - Nightly on a self-hosted runner with SU2 installed
  - On every release-candidate tag
- Cache the mesh on a content hash of the contour `.csv`.
- Cache the SU2 solution likewise. Re-run only when contour or solver
  config changes.

### 10.7 Test gate (Phase 11)

The `e10_l80_g14` canonical case (ε=10, length_pct=80, γ=1.4) passes
all three tolerance checks. `bell_nozzle_contour(method='rao_variational_moc')`
emits `reliability = CFD_CHECKED` only when:
- The chart benchmark test has passed for this geometry, AND
- The CFD comparison has been run within the last release and is
  cached as passing, AND
- The current run's contour-hash matches the cached CFD-validated
  contour-hash.

Otherwise reliability stays at `BENCHMARK_VALIDATED` or lower.

---

## 11. Updated reliability ladder

After phases 1-11, the `ContourReliability` enum at
`rao_variational.py:98` should mean:

| Level | Required gates passed |
|---|---|
| `GEOMETRIC_APPROXIMATION` | Anything that doesn't satisfy `MOC_COMPATIBLE`. |
| `MOC_COMPATIBLE` | Phase 1-2: no post-processing, no endpoint forcing, axisymmetric C±/Q± residuals < tol. |
| `RAO_VARIATIONAL_RESIDUAL_SOLVED` | Phase 3-6: Rao algebraic stationarity + left-Mach-line + mass closure + valid-region + coupled wall. |
| `BENCHMARK_VALIDATED` | Phase 7: chart sweep agrees with NASA RP-1104 within ±1.5° RMS. |
| `CFD_CHECKED` | Phase 11: SU2 CFD comparison within 0.5% Cf, 3% wall p, 1% exit M. |
| `EXPERIMENTALLY_VALIDATED` | Future work; requires hot-fire data. Not addressed in this plan. |

`hardware_qualified = True` requires `EXPERIMENTALLY_VALIDATED` and
explicit sign-off in materials, manufacturing, and inspection
metadata, which is out of scope for the solver/contour code itself.

---

## 12. Real-gas considerations no longer "out of scope"

The original §8 marked the following as out of scope. They are now in
scope as Phase 9-11:

- Kliegel-Levine transonic kernel → Phase 9.
- Real-gas / variable-γ / CEA-frozen and shifting-equilibrium MOC →
  Phase 10.
- Supersonic CFD cross-check → Phase 11.

What remains genuinely out of scope:

- **Finite-rate chemistry MOC** (not Bray sudden-freezing, but a full
  species-ODE-along-streamline approach as in TDK). This requires a
  CHEMKIN-class reaction-rate library and is a separate workstream.
- **Hot-fire experimental validation**, which requires hardware,
  test stands, and instrumented runs. Out of scope for any software
  effort.
- **Side-load / separation transient analysis** under altitude excursions.
  Östlund's thesis (`fulltext01.pdf`) treats this; a future Phase 12
  could add a separation-onset predictor wired to the validated
  contour, but it is design-margin work, not solver math.

---

*Document owner:* (you). Update each phase's checkboxes as work lands.
Do not delete the "what the other model got wrong" section — it is the
canonical reason future contributors should not re-do this same audit.
