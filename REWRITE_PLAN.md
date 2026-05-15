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

**NASA reference code — `Three-Dimensional-Nozzle-Design-Code-master/`
is now in this repo.** It is the JHU/APL "MOC_Grid_BDE" toolset
(Rice, T., *2D and 3D Method of Characteristic Tools for Complex
Nozzle Development*, JHU/APL Report RTDC-TPS-481, 2003 — that is your
`propulsion_texts/20030067852.pdf`, by the way). The MOC core is in
plain C++ (3754 lines in `MOC_GridCalc_BDE.cpp`); MFC dependencies are
limited to GUI wrappers (`StdAfx.h`, `Chart.h`, `dummyStruct.h`).
The class exposes exactly the topology we are missing:

| NASA function | What it computes |
|---|---|
| `CalcInitialThroatLine` (line 2805) | Initial data line TT' on the throat arc |
| `KLThroat` (line 3103) | **Full Kliegel-Levine 1969 in toroidal coords through 3rd order** — explicit polynomial coefficients for both AXI and TWOD |
| `Sauer` (line 3054) | Modified Sauer transonic (axi + planar) — what your current `'hall'` actually is |
| `CalcMdotBD(j, xD)` (line 1436) | Mass flow integrated from wall down to point D on RRC `j` — exactly the integral §2.D needs |
| `CalcLRCDE` (line 1472) | The LRC DE construction (i.e. the Rao control surface), secant-iterated on `xD` so mass(BD)=mass(DE) |
| `FindPointE` (line 1764) | Given D and the BD mass flow, integrate forward along the LRC until mass matches |
| `CalcDE` (line 3638) | Outer DE construction driver |
| `CalcBDERegion` (line 3258) | The BDE strip (region inside the kernel triangle) |
| `CalcWallContour` (line 1167) | Final wall extraction from BDE / streamline trace |
| `SetThetaB` (line 3718) | Secant iteration on initial expansion angle θ_B |
| `RungeKuttaFehlberg` (line 3458) | RK4(5) integration along characteristics |
| `OutputTDKRAODataFile` | Writes `rao.dat` in TDK-compatible format |

Sample outputs in `MOC_Grid_BDE/outputs_M3.5Perf/` (Mach 3.5 perfect
nozzle, axi, γ=1.4, R*=1, P0=500 psia) include `wall.out`,
`center.out`, `summary.out`, `MOC_Grid.plt`, `MOC_SL.plt`,
`TT'BF_Kernel.out`, `BFE_Kernel.out`, `ThetaB.out`, `rao.dat`,
`axis_i.out`, `wall_i.out`, `UncroppedKernel.out`, `LastKernel.out` —
these become **immediate, free, bit-comparable regression targets**.

This NASA code is the single biggest piece of leverage available to
this project. Phase 12 (§13 below) lays out how to use it.

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
| 9 | `test_kl_throat_matches_nasa_axis_mach`, `test_kl_reduces_to_1d_in_large_Rc_limit`, `test_kl_vs_cuffel_back_massier_1969` | `tests/test_transonic_kernel.py` (new) |
| 10 | `test_constant_gamma_matches_existing_prandtl_meyer`, `test_frozen_cea_recovers_constant_gamma_in_limit`, `test_shifting_equilibrium_bartz_higher_than_chamber_frozen` | `tests/test_thermo.py` (new) |
| 11 | `test_su2_axisymmetric_cf_within_half_percent` | `tests/test_cfd_su2.py` (new, `@cfd`) |
| 12 | `test_legacy_io_parses_M3.5Perf_outputs`, `test_python_port_matches_nasa_wall_out_rms_1e3`, `test_topology_objects_present_after_solve` | `tests/test_legacy_io.py`, `tests/test_nasa_port.py` (new) |
| 13 | `test_plot_characteristic_net_smoke`, `test_plot_geometry_raw_vs_export_differs`, `test_plot_nasa_overlay_runs` | `tests/test_plotting.py` (new) |

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

**Pre-Phase 1 (one evening):** do the NASA legacy audit (§13.1) and
the topology map (§13.2). These cost almost nothing and they unblock
everything: you stop guessing what "B, BF, D, BD, DE, E" mean and
start matching every later phase against a known reference.

If you only have a weekend, do Phase 1 (honest naming + reliability gates +
no-silent-postprocess) and Phase 2 (axisymmetric compatibility residuals).
Those two together neutralize the most dangerous failure mode (silent
emission of a non-Rao contour) without requiring the algebraic Rao math
to land yet. Also land §14.1 (the characteristic-net plot) — without
visual diagnostics, every later phase ships blind.

If you have a week, add Phase 3 (algebraic stationarity + left-Mach-line),
Phase 4 (mass closure — port `CalcMdotBD` / `FindPointE` from the NASA
C++ as the reference implementation, see §13.7), and §14.2 (Mach-colored
flowfield plot).

If you have a month, do everything through Phase 7 and ship the chart
benchmark. At that point — and only at that point — the
`rao_variational_moc` path is publishable as a Rao-method contour
generator. Until then, the contour is a "Rao-flavored variational guess"
and should be labeled as such (which the current `design_status =
"experimental_rao_variational_moc_bvp"` already does, but most users
will not read).

The NASA repo (§13) cuts most of these timelines in half. The
`outputs_M3.5Perf/` reference case is a free oracle — every port can
be compared to it without writing a benchmark from scratch.

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
u/a* = 1 +  u1/(R+1) + (u1 + u2)/(R+1)²  + (u1 + 2u2 + u3)/(R+1)³
v/a* = √((γ+1)/(2(R+1))) · [ v1/(R+1) + (1.5 v1 + v2)/(R+1)² + ((15/8) v1 + 2.5 v2 + v3)/(R+1)³ ]
```

where `u1, u2, u3, v1, v2, v3` are polynomial in `(y, z; γ)` with
`y = r/Rt` and `z = x · √(2 Rc/(γ+1))` (toroidal coordinate, Hall Eq. 12).

**Use the NASA `KLThroat` source directly.** The full coefficient
expansion is in `Three-Dimensional-Nozzle-Design-Code-master/MOC_Grid_BDE/MOC_GridCalc_BDE.cpp:3103-3178`.
For axisymmetric flow, the coefficients literally are:

```cpp
// From NASA MOC_GridCalc_BDE.cpp, KLThroat, AXI branch, lines 3122-3135
u[1] = y*y/2 - 0.25 + z;
v[1] = y*y*y/4 - y/4 + y*z;
u[2] = (2*G + 9)*y*y*y*y/24 - (4*G + 15)*y*y/24 + (10*G + 57)/288
     + z*(y*y - 5/8) - (2*G - 3)*z*z/6;
v[2] = (G + 3)*y*y*y*y*y/9 - (20*G + 63)*y*y*y/96 + (28*G + 93)*y/288
     + z*((2*G + 9)*y*y*y/6 - (4*G + 15)*y/12) + y*z*z;
u[3] = (556*G*G + 1737*G + 3069)*y*y*y*y*y*y/10368
     - (388*G*G + 1161*G + 1881)*y*y*y*y/2304
     + (304*G*G + 831*G + 1242)*y*y/1728
     - (2708*G*G + 7839*G + 14211)/82944
     + z*((52*G*G + 51*G + 327)*y*y*y*y/34
          - (52*G*G + 75*G + 279)*y*y/192
          + (92*G*G + 180*G + 639)/1152)
     + z*z*(-(7*G - 3)*y*y/8 + (13*G - 27)/48)
     + (4*G*G - 57*G + 27)*z*z*z/144;
v[3] = ...  // see lines 3131-3135 — note: line 3133 contains a typo
            // in the NASA source (`*` where `+` is intended); cross-
            // check against Hall 1962 Eq. 5.7 or Kliegel-Levine 1969
            // Table 1 when porting.
```

**Critical: the NASA file at line 3133 contains a typo** — the
expression for `v[3]` has `*(388*G*G + 1161*G + 1181)*y*y` where it
should be `+ (388*G*G + 1161*G + 1881)*y*y/576`. Cross-check the
TWOD branch on line 3157 for the same typo (it appears there too).
Use Kliegel-Levine 1969 Table 1 or Östlund §3 as the disambiguator.
**Document the typo correction in the docstring of the Python port.**

Secondary references for cross-validation:

- **Zucrow & Hoffman, *Gas Dynamics Vol. 2*, Ch. 16** — full coefficient
  table, primary textbook source.
- **Östlund, KTH thesis 2002** (your `fulltext01.pdf`) — reproduces the
  Kliegel-Levine coefficients in §3.
- **Kliegel & Levine 1969, AIAA J. 7(7) 1375-1378**, DOI 10.2514/3.5355.
  Open access: <https://arc.aiaa.org/doi/pdf/10.2514/3.5355>
- **Kliegel NTRS AED-R-71-10**:
  <https://ntrs.nasa.gov/api/citations/19710024785/downloads/19710024785.pdf>
- **Sivells AEDC nozzle code (DTIC ADA062944)** — FORTRAN parent of
  most modern MOC codes.

A reference Python port to diff against:
<https://github.com/noahess/conturpy>.

The NASA `Sauer` (lines 3054-3098) is **also worth porting** as a
fast subsonic fallback for the throat plane. It is what NASA uses
when KLThroat fails (subsonic initial line, see line 3173). Keep the
current `'hall'` method renamed to `'sauer_modified'` since that is
what it actually implements (look at `moc.py:371-374` — the formulas
match NASA's `Sauer` lines 3073-3076, AXI branch, with simplified
coefficients).

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

## 11. Phase 12 — NASA MOC_Grid_BDE integration (the canonical reference)

The NASA Three-Dimensional-Nozzle-Design-Code from JHU/APL gives us
something rare: a **published, end-to-end Rao optimum-thrust MOC
implementation in source form, with sample outputs**. Every later
phase of this rewrite (algebraic stationarity, mass closure, valid
region, coupled wall) becomes much cheaper to validate when we can
compare directly against this reference. This whole section is the
prior model's "Phase 1-12 legacy audit" plan, folded into the
overall RaoRocketSim rewrite plan and adjusted to match what is
actually in the NASA repo.

### 11.1 Pre-work: file inventory and topology map (do this first)

Deliverables:

- `docs/nasa_jhu_file_list.txt` — output of
  `find Three-Dimensional-Nozzle-Design-Code-master -type f`.
- `docs/legacy_code_audit.md` — for each of the three NASA programs
  (`MOC_Grid_BDE`, `STT2001`, `3D_MOC`), record:
  - One-paragraph purpose statement (`MOC_Grid_BDE` is the only one
    relevant to our 2D-axisymmetric path right now).
  - File list grouped into: numerical core, MFC GUI wrappers, sample
    outputs, project files (`.dsw`, `.rc`, `.odl`).
  - Build dependencies (Visual Studio 6/2003 era MFC).
  - Output file list and meaning of each file (`wall.out`, `center.out`,
    `MOC_Grid.plt`, `TT'BF_Kernel.out`, `BFE_Kernel.out`, `rao.dat`,
    `ThetaB.out`, `Summary.plt`, `axis_i.out`, `wall_i.out`,
    `UncroppedKernel.out`, `LastKernel.out`).

- `docs/rice_jhu_moc_topology_map.md` — the most important deliverable
  in this section. A table mapping JHU/APL terminology to RaoRocketSim
  objects. Initial fill-in (refine after reading Rice 2003 / your
  `20030067852.pdf`):

  | Rice/JHU term | Meaning | NASA function | RaoRocketSim status |
  |---|---|---|---|
  | TT' | Initial data line on throat arc | `CalcInitialThroatLine`, `KLThroat`, `Sauer` | `moc.approximate_starting_line` (weak; replace per Phase 9) |
  | TB | Initial expansion arc, throat → B | `CalcRRCsAlongArc`, `CalcArcWallPoint` | partial; kernel seed only |
  | B | End of expansion region (last LRC origin on wall arc) | `iBD, jBD` members | not explicitly represented |
  | BF | Final RRC of the kernel triangle | array slice in `CalcLRCDE` | implicit in CE start |
  | D | Selected point on BF for Rao construction | secant variable in `CalcLRCDE` | **missing**; currently `x_ce[0] = Nx + 0.05·(Ln−Nx)` heuristic |
  | BD | Wall-to-D curve, used for mass-flow target | `CalcMdotBD(j, xD)` | **missing**; currently `_target_mdot = ρV*·A_t` quasi-1D |
  | DE | Left-running characteristic D→E, the control surface | `CalcLRCDE`, `FindPointE` | our `ControlSurface`, but topology not verified |
  | E | Nozzle exit point | endpoint of `FindPointE` integration | fixed at `(Ln, Re)` |
  | B→E streamline | Final wall contour | `CalcWallContour` | `_construct_wall_from_ce` (sequential; replace per §2.F) |
  | θ_B iteration | Outer secant on initial expansion angle | `SetThetaB` | none — fixed input |

The Rice 2003 report (= your `propulsion_texts/20030067852.pdf`)
explicitly states that **only one combination of initial expansion
angle θ_B and point D satisfies all the constraints** for a Rao
optimum-thrust nozzle solution. That is the topological reason your
current CE-then-wall sequential construction can close *locally*
while the forward characteristic net fails *globally* — your D is
heuristic and your θ_B is not iterated.

### 11.2 Phase 12.1 — Legacy I/O parsers (before any C++ binding)

Build Python parsers for the NASA output formats **before** trying
to compile or wrap the C++. This buys 80% of the reference value at
20% of the cost.

New module `raosim/legacy_io.py` with:

```python
def parse_wall_out(path: str) -> WallTable:
    """
    Parse NASA wall.out:
        header: i  j  X/R*  R/R*  mach  theta(deg)  Pressure, psia
        rows:   tab-separated floats
    Returns a structured object with arrays.
    """

def parse_center_out(path: str) -> CenterlineTable:
    """
    Parse NASA center.out:
        header line, then comment header, then tab-separated:
        J  X/R*  R/R*  Mach  Pres  Temp  Rho  Theta  Gamma  MassFlow
    """

def parse_rao_dat(path: str) -> RaoDat:
    """
    Parse NASA rao.dat:
        3 columns, space-or-tab separated.  Likely (X/R*, R/R*, theta_deg)
        — verify against MOC_GridCalc_BDE_IO.cpp::OutputTDKRAODataFile
        before relying.
    """

def parse_moc_grid_plt(path: str) -> MOCGridPlot:
    """
    Parse NASA MOC_Grid.plt — Tecplot ASCII format with
    VARIABLES, ZONE I=, J=, headers.  Returns a 2D mesh.
    """

def parse_kernel_out(path: str) -> KernelTable:
    """
    Parse NASA TT'BF_Kernel.out / BFE_Kernel.out / UncroppedKernel.out
    / LastKernel.out.  These give the kernel triangle topology
    explicitly.
    """

def parse_thetaB_out(path: str) -> ThetaBHistory:
    """
    Parse ThetaB.out — the secant-method iteration history for
    the outer initial-expansion-angle solve.  Each row is one
    iteration: theta_B [deg], error.
    """

def parse_summary_out(path: str) -> SummaryReport:
    """
    Parse summary.out — solver summary including nozzle type,
    geometry, design parameter, gamma, all derived constants,
    and the final thrust coefficients (Cf, Cfg, CD, C*).
    """
```

Each parser returns a small dataclass with numpy arrays and the
provenance dict (file path, NASA version stamp from `summary.out`
header).

Test gate: `tests/test_legacy_io.py` parses every file in
`MOC_Grid_BDE/outputs_M3.5Perf/`, validates row counts, and
asserts that the first few rows match hand-typed truth values.

### 11.3 Phase 12.2 — Comparison harness

Deliverable: `scripts/compare_nasa_reference.py`

```python
def main():
    # 1. Load NASA outputs_M3.5Perf via raosim.legacy_io
    nasa_wall = parse_wall_out("...outputs_M3.5Perf/wall.out")
    nasa_center = parse_center_out("...outputs_M3.5Perf/center.out")
    nasa_summary = parse_summary_out("...outputs_M3.5Perf/summary.out")

    # 2. Build an equivalent RaoSolverConfig from NASA's inputs.
    #    Mach 3.5 perfect axi nozzle, γ=1.4, P0=500 psia, T0=530 R, MW=28.96,
    #    R* = 1 in, Rup/R* = Rdown/R* = 1.
    config = RaoSolverConfig(
        Rt = 0.0254,            # 1 inch in meters
        epsilon = mach_to_epsilon(3.5, 1.4),
        gamma = 1.4,
        # ... and configure the BVP to mimic NASA's "PERFECT" nozzleType
    )
    sol = solve_rao_bvp(config)

    # 3. Diff
    rms_x = rms(nasa_wall.x_over_Rstar - sol.wall_export[:, 0] / Rt)
    rms_r = rms(nasa_wall.r_over_Rstar - sol.wall_export[:, 1] / Rt)
    rms_M = rms(nasa_wall.mach - interp(sol.wall_x, sol.wall_M, nasa_wall.x))
    rms_theta = rms(nasa_wall.theta - interp(sol.wall_x, sol.wall_theta, nasa_wall.x))
    rms_p = rms(nasa_wall.pressure - interp(...))

    # 4. Emit a diff report + plots into debug_outputs/nasa_comparison/
    print(f"Wall x RMS: {rms_x:.4g}")
    print(f"Wall r RMS: {rms_r:.4g}")
    print(f"Wall M RMS: {rms_M:.4g}")
    print(f"Wall theta RMS: {rms_theta:.4g}")
    print(f"Wall pressure RMS: {rms_p:.4g}")
```

Add `debug_outputs/nasa_comparison/.gitkeep` so the directory persists
in the repo without committing actual diff artifacts.

The **first goal is to expose differences, not to make them small**.
Do not gate any reliability flag on this comparison until Phases 1-6
have landed.

### 11.4 Phase 12.3 — Port-or-bind decision

Three options. Choose one, document it in `docs/legacy_strategy.md`:

**Option A: Reference-only, no binding (recommended starting point).**
Treat the NASA outputs as ground-truth oracle. Use the comparison
harness from §11.3 as the regression test. Re-implement the algorithm
phase-by-phase in Python with the NASA source code visible as you
write each piece. This is what the prior model called "Option A —
No binding, just reference outputs." Risk: low. Time: bounded.

**Option B: Subprocess runner.** If the NASA executable can be
built on a Windows VM, wrap it in `raosim/legacy_runner.py`:

```python
def run_legacy_moc_case(case_dict, work_dir) -> dict:
    """
    Write NASA-format input file, invoke MOC_Grid_BDE.exe, parse outputs.
    Requires Windows or wine. Returns parsed outputs.
    """
```

Use case: generating new reference cases beyond the shipped
`outputs_M3.5Perf/`. Risk: medium (MFC build, GitHub issue thread
on the NASA repo mentions difficulty locating `main`; the project
is MFC dialog-driven). Time: a few days of build-system pain.

**Option C: pybind11 binding of the numerical core.** Extract the
plain-C++ subset from `MOC_GridCalc_BDE.{h,cpp}`, `engineering_constants.hpp`,
`Vector.hpp`, `Matrix.hpp` into a freestanding library. Stub or replace
`StdAfx.h`, `dummyStruct.h`, `Chart.h`. Write a CMakeLists.txt.
Wrap `MOC_GridCalc` with pybind11. Distribute as a Python extension.

The NASA class is actually amenable to this: `MOC_GridCalc` itself
has no MFC inside the math — calls to `AfxMessageBox` are limited
to error reporting (e.g. line 3173, "Initial Data Line is subsonic")
and can be replaced with C++ exceptions. The `dummyStruct.h` header
defines a plain POD-with-array-members result type — replace with a
proper `struct`. Risk: high (real C++ work, ~80-200 hours). Time:
a few weeks. Payoff: the C++ code itself becomes a Python-importable
module, the way the user originally asked.

**Recommended path:** A first, then C if/when the Python port is fully
validated and you want production-grade Rao without re-deriving every
piece. B is not worth the effort; C dominates B as soon as A is in
hand.

### 11.5 Phase 12.4 — Algorithm port (Option A in detail)

For each NASA function, write a Python equivalent in
`raosim/nasa_moc.py`. Keep the same function name (with `snake_case`).
Add an inline comment at the top of each function:

```python
def calc_mdot_bd(grid: MOCGrid, j: int, xD: float) -> float:
    """
    Mass flow from wall down to point D on RRC j.

    Port of MOC_GridCalc::CalcMdotBD
    (MOC_GridCalc_BDE.cpp:1436-1467, NASA/JHU 2003).
    """
```

Priority order:

1. **Sauer + KLThroat** — port lines 3054-3178. Verify against the
   axis-Mach number in `outputs_M3.5Perf/wall.out` first column.
   Test gate: `abs(M_python_at_axis - 1.17779) < 1e-4`.
2. **CalcMu, CalcA, CalcB, Calcb, CalcR, CalcRStar, lDyDx, rDyDx,
   TanAvg, MM, CalcPMFunction** — small math helpers (lines
   2957-3052). Trivial ports; do as a batch.
3. **CalcIsentropicP_T_RHO** — already exists in `gas_dynamics.py`
   but the NASA version takes different units (psia + Rankine).
   Add a units-adapter, not a re-port.
4. **CalcRRCsAlongArc + CalcArcWallPoint** — the TB initial-
   expansion arc construction (lines 1030, 835). This is the kernel
   build. Port carefully.
5. **CalcMdotBD** (line 1436) — small (32 lines). The integral
   itself is just a piecewise-linear interpolation of pre-computed
   `massflow[i][j]` values. The work is the mass-flow accumulation
   loop in `CalcMassFlowAndThrustAlongMesh` (line 3183).
6. **CalcLRCDE + FindPointE + RungeKutta + RungeKuttaFehlberg +
   Deriv** (lines 1472, 1764, 3414, 3458, 3514) — the heart of the
   Rao mass-closure inner loop. This is what replaces the
   `_target_mdot` quasi-1D placeholder per §2.D.
7. **CalcDE + CalcBDERegion + CalcRemainingMesh + CalcWallContour
   + CropNozzleToLength** (lines 3638, 3258, 1122, 1167, 1341) —
   the outer Rao construction.
8. **SetThetaB** (line 3718) — outer secant on θ_B. This is what
   replaces the fixed `thetaN_guess_deg` in `RaoSolverConfig`.
9. **CalcContouredNozzle** (line 241) — the top-level driver.

After this port, `raosim/nasa_moc.py` contains a Python class
`MOCGridCalc` with the same public API as the C++ version. The
existing `solve_rao_bvp` becomes a thin wrapper that builds an
`MOCGridCalc`, calls `create_moc_grid()`, and shapes the result
into a `RaoSolution`. The variational-residual block from earlier
phases becomes an optional refinement layer on top of this.

### 11.6 Phase 12.5 — Tests against NASA ground truth

For each port, add a regression test in `tests/test_nasa_port.py`:

```python
@pytest.mark.parametrize("case", ["M3.5Perf"])
def test_nasa_wall_match(case):
    nasa_dir = REPO_ROOT / "Three-Dimensional-Nozzle-Design-Code-master" \
                          / "MOC_Grid_BDE" / f"outputs_{case}"
    nasa_wall = parse_wall_out(nasa_dir / "wall.out")
    py_solution = run_python_port_for_case(case)
    assert rms(py_solution.wall_x_over_Rstar - nasa_wall.x_over_Rstar) < 1e-3
    assert rms(py_solution.wall_M - nasa_wall.M) < 1e-3
```

Initial tolerances should be loose (e.g. `1e-2`). Tighten phase by
phase. The eventual target is `1e-4` relative on every wall quantity,
which is bit-comparable given that NASA uses single-precision in
some places.

### 11.7 Phase 12.6 — Topology objects in the Python codebase

After porting, formalize the topology in `raosim/moc_topology.py`:

```python
@dataclass
class RaoTopology:
    TT_prime: list[CharPoint]    # initial data line
    B: CharPoint                  # end-of-expansion wall point
    BF: list[CharPoint]           # final RRC of the kernel
    D: CharPoint                  # selected point on BF
    BD: list[CharPoint]           # mass-flow curve (wall to D on j_B)
    DE: list[CharPoint]           # control surface (LRC from D to E)
    E: CharPoint                  # nozzle exit point
    streamline_BE: list[CharPoint]  # wall contour from B to E
    theta_B: float                # outer-loop converged value
    mass_BD: float
    mass_DE: float                # must equal mass_BD at convergence
```

This is the "explicit objects" deliverable from the prior model's
Phase 9. After this lands, `_construct_wall_from_ce` is deleted
and the wall is the `streamline_BE` field of the topology object.

### 11.8 Phase 12 test gate

- All NASA output files in `outputs_M3.5Perf/` parse without error.
- `compare_nasa_reference.py` runs and emits a difference report
  (no tolerance yet).
- After §11.5 ports land, RMS wall x/r/M/θ/p agreement with NASA
  is within `1e-3` relative.
- The `RaoTopology` object is the new internal representation; no
  consumer of `solve_rao_bvp` reads `control_surface`, `wall_raw`,
  etc. directly any more.

---

## 12. Phase 13 — Visualization and flowfield diagnostics

The existing `raosim/plotting.py` (166 lines) has nozzle-2D / nozzle-3D /
curvature plots. It does **not** have the diagnostic flowfield plots
needed to debug the BVP / MOC / Rao construction. Add them.

Without visual diagnostics, every phase of this rewrite ships blind —
numeric residuals and "converged: True/False" are abstract. A plot
turns "row 12 has cplus_rms = 3e-2" into "I can see row 12 folding
into row 11 right at the throat exit." That is the difference between
a half-day debug session and a week.

### 12.1 Required plots

Add to `raosim/plotting.py`:

1. **`plot_nozzle_geometry(solution)`** — wall contour, axis, throat,
   inflection N, exit E, with the convergent + throat arcs visible
   separately so the seam at N is inspectable.

2. **`plot_characteristic_net(solution)`** — wall + CE + kernel start
   line + every characteristic in `solution.characteristic_net`. Use
   separate line styles for C+ and C− families (solid vs dashed).
   This is the **first plot to land** — single-handedly catches:
   crossing characteristics, fold-over rows, CE/wall mismatch,
   blown-up rows, weird kernel topology.

3. **`plot_flowfield_mach(solution)`** — scatter every node in the
   characteristic net colored by Mach number. Same axes overlay as
   `plot_characteristic_net`.

4. **`plot_flowfield_pressure(solution, gamma)`** — same, colored
   by `p/p0` from `isentropic_pressure_ratio(M, gamma)`. Verifies
   monotonic decrease downstream.

5. **`plot_flowfield_theta(solution)`** — flow angle field. Verifies
   exit alignment.

6. **`plot_wall_distributions(solution)`** — three subplots stacked:
   x vs wall Mach, x vs wall θ, x vs wall p/p0. This is what
   propulsion engineers actually read.

7. **`plot_exit_plane(solution)`** — r vs M(r), r vs θ(r), r vs p(r)
   at the exit plane. Catches non-uniform exit profile, residual
   turning, over/under-expansion.

8. **`plot_net_diagnostics(solution)`** — characteristic net with
   problematic links highlighted. Inputs:
   - link RMS residuals from `RaoResidualReport`
   - characteristic crossings from `check_characteristic_crossing`
   - bad rows (where wall-tangency RMS or compatibility RMS exceeds
     row-local tolerance)
   Color the worst 5% of links in red; print the offending link
   indices to stdout.

9. **`plot_topology(rao_topology)`** — annotated overlay of TT', B,
   BF, D, BD, DE, E onto the wall + characteristic net. Only valid
   after Phase 12.7 lands.

10. **NASA reference overlay plots** (`plot_nasa_overlay(solution,
    nasa_outputs_dir)`). Loads NASA's `wall.out` / `center.out` via
    `legacy_io` and overlays them as scatter markers on
    `plot_wall_distributions` / `plot_flowfield_mach`. Direct visual
    diff against the JHU/APL reference.

### 12.2 Implementation outline

```python
# raosim/plotting.py

def plot_characteristic_net(solution, ax=None, show_families=True):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    # Wall (raw, NOT export — see §12.3 below)
    wall = solution.wall_raw
    ax.plot(wall[:, 0], wall[:, 1], color="black", linewidth=2, label="wall (raw)")

    # CE
    ce = solution.control_surface
    ax.plot(getattr(ce, "x", []), ce.r, "--", label="control surface CE")

    # Kernel starting line
    if solution.kernel_points:
        xk = [p.x for p in solution.kernel_points]
        rk = [p.r for p in solution.kernel_points]
        ax.plot(xk, rk, "o-", markersize=3, label="kernel / starting line")

    # Characteristic rows
    for row in solution.characteristic_net:
        pts = row.all_points()
        ax.plot([p.x for p in pts], [p.r for p in pts],
                linewidth=0.5, color="C0", alpha=0.5)

    ax.axhline(0.0, color="grey", linewidth=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("r")
    ax.set_title("Characteristic net (raw)")
    ax.set_aspect("equal", "box")
    ax.legend(loc="best", fontsize=8)
    return ax


def plot_flowfield_mach(solution, ax=None, cmap="viridis"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    xs, rs, Ms = [], [], []
    for row in solution.characteristic_net:
        for p in row.all_points():
            xs.append(p.x); rs.append(p.r); Ms.append(p.M)

    sc = ax.scatter(xs, rs, c=Ms, s=12, cmap=cmap)
    wall = solution.wall_raw
    ax.plot(wall[:, 0], wall[:, 1], color="black", linewidth=2)
    ax.axhline(0.0, color="grey", linewidth=0.5)
    ax.set_aspect("equal", "box")
    ax.set_xlabel("x")
    ax.set_ylabel("r")
    ax.set_title("Mach number field")
    plt.colorbar(sc, ax=ax, label="Mach")
    return ax
```

Once `RaoNetLink` topology lands (Phase 6 / 12.7), the
`plot_characteristic_net` body should iterate over explicit
`(parent, child, family)` links instead of `row.all_points()`,
because naive row-adjacency lies once the net deforms.

### 12.3 The raw-vs-export trap

**Plots must default to raw, not exported, geometry.** The whole
point of §2.F / §6 is to remove silent endpoint forcing and
monotonic cleanup from the wall before plotting. If
`plot_nozzle_geometry` falls through to `solution.wall_export`,
it will hide exactly the kind of mismatch you need to see during
debugging.

Convention:

```python
def plot_nozzle_geometry(solution, *, geometry="raw", ax=None):
    if geometry == "raw":
        wall = solution.wall_raw
        label = "wall (raw, BVP output)"
    elif geometry == "export":
        wall = solution.wall_export
        label = "wall (export, post-processed)"
    else:
        raise ValueError(...)
    # ... plot
    ax.set_title(f"Nozzle geometry — {geometry}")
```

Default to `geometry="raw"`. Plotting export geometry without explicit
opt-in repeats the exact silent-failure mode this rewrite is
eliminating.

### 12.4 Tests

`tests/test_plotting.py`:

1. Smoke tests — every plotting function runs to `Figure.savefig`
   without raising on the (ε=10, length_pct=80) reference case.
2. Geometry switch test — `plot_nozzle_geometry(sol, geometry="raw")`
   draws different content from `plot_nozzle_geometry(sol, geometry="export")`
   when the wall has been post-processed.
3. NASA overlay test — `plot_nasa_overlay` runs without raising on
   `outputs_M3.5Perf/` paths.

Use `matplotlib.testing.compare.compare_images` against a small set
of baseline PNGs for visual regression, but mark them
`@pytest.mark.image` and skip in CI unless explicitly requested
(image comparisons are flaky in headless CI).

### 12.5 Phase 13 test gate

- All ten plot functions render without error on the reference case.
- `plot_net_diagnostics` correctly highlights at least one bad
  link in a deliberately mis-converged solver run
  (`max_nfev=0` initial-residual-only solution).
- A `make plots` shortcut (or `python -m raosim.plotting <case>`)
  regenerates the standard plot set into `debug_outputs/plots/`.

---

## 13. Updated reliability ladder

After phases 1-13, the `ContourReliability` enum at
`rao_variational.py:98` should mean:

| Level | Required gates passed |
|---|---|
| `GEOMETRIC_APPROXIMATION` | Anything that doesn't satisfy `MOC_COMPATIBLE`. |
| `MOC_COMPATIBLE` | Phase 1-2: no post-processing, no endpoint forcing, axisymmetric C±/Q± residuals < tol. |
| `RAO_VARIATIONAL_RESIDUAL_SOLVED` | Phase 3-6: Rao algebraic stationarity + left-Mach-line + mass closure + valid-region + coupled wall, AND topology objects (`B, BF, D, BD, DE, E`) present. |
| `NASA_REFERENCE_MATCHED` *(new)* | Phase 12: RMS wall x/r/M/θ/p agreement with NASA `outputs_M3.5Perf/` better than 1e-3 relative. **This is the new gold standard for "matches the published Rao algorithm."** Use NASA outputs, not RP-1104 chart angles — the chart is interpolated, the NASA outputs are direct. |
| `BENCHMARK_VALIDATED` | Phase 7: chart sweep over (ε, length_pct) agrees with NASA RP-1104 (your `19830016278.pdf`) within ±1.5° RMS on θ_N and θ_E. |
| `CFD_CHECKED` | Phase 11: SU2 CFD comparison within 0.5% Cf, 3% wall p, 1% exit M. |
| `EXPERIMENTALLY_VALIDATED` | Future work; requires hot-fire data. Not addressed in this plan. |

Add `NASA_REFERENCE_MATCHED` to the enum at `rao_variational.py:98`
between `RAO_VARIATIONAL_RESIDUAL_SOLVED` and `BENCHMARK_VALIDATED`.
This codifies that the NASA repo is now treated as the canonical
reference implementation — a contour matching it bit-comparably is
trustworthy in a way that "agrees with a chart to ±2°" is not.

`hardware_qualified = True` requires `EXPERIMENTALLY_VALIDATED` and
explicit sign-off in materials, manufacturing, and inspection
metadata, which is out of scope for the solver/contour code itself.

### 13.1 Methodological note: don't contaminate the chart benchmark

The current code seeds the BVP with `thetaN_guess_deg=30.0`
(`rao_variational.py:127`) and `_lookup_theta_n` from
`rao_optimizer.py:44` reads the NASA SP-8120 chart. Once Phase 3
lands and the BVP solves for θ_N from the Rao stationarity equations
themselves, **the chart values must remain initial guesses only, not
residuals**.

Add a `RaoSolverConfig.angle_boundary_mode` field:

- `"free"` (default) — chart angles seed `_initial_ce_from_kernel`
  but never appear in the residual stack.
- `"chart_soft"` (debug only) — a tiny extra residual
  `(theta_N_solver - theta_N_chart)` with weight 1e-3 to help
  diagnose pathological convergence.
- `"chart_hard"` (deprecated) — the current implicit behavior when
  `thetaN_guess_deg` is used as an upper bound; emit a deprecation
  warning when this is selected.

The Phase 7 chart benchmark **must** use `angle_boundary_mode="free"`
or it is circular.

---

## 14. Real-gas, CFD, and KL considerations no longer "out of scope"

The original §8 of the first plan marked the following as out of scope.
They are all now in scope:

- Kliegel-Levine transonic kernel → **Phase 9** (with NASA's `KLThroat`
  as a direct port target — see §8.1).
- Real-gas / variable-γ / CEA-frozen and shifting-equilibrium MOC →
  **Phase 10**.
- Supersonic CFD cross-check → **Phase 11** (SU2 axisymmetric Euler
  against NASA RP-1104 = your `19830016278.pdf`).
- NASA Three-Dimensional-Nozzle-Design-Code integration → **Phase 12**
  (port + reference outputs as canonical regression targets).
- Visualization / debugging plots → **Phase 13**.

What remains genuinely out of scope:

- **Finite-rate chemistry MOC** (not Bray sudden-freezing, but a full
  species-ODE-along-streamline approach as in TDK). This requires a
  CHEMKIN-class reaction-rate library and is a separate workstream.
- **Hot-fire experimental validation**, which requires hardware,
  test stands, and instrumented runs. Out of scope for any software
  effort.
- **Side-load / separation transient analysis** under altitude excursions.
  Östlund's thesis (`fulltext01.pdf`) treats this; a future Phase 14
  could add a separation-onset predictor wired to the validated
  contour, but it is design-margin work, not solver math.
- **3D / scramjet asymmetric nozzles.** The NASA `3D_MOC` and `STT2001`
  programs in `Three-Dimensional-Nozzle-Design-Code-master/` handle
  3D non-axisymmetric flow and streamline tracing for scramjet inlets.
  These are out of scope for RaoRocketSim's axisymmetric rocket bell
  pipeline, but the `STT2001` streamline tracer might be useful
  long-term if you ever extend to dual-bell or aerospike geometries.
  Note for the future, not for this rewrite.
- **Wrapping the NASA MFC GUI itself.** Option C (pybind11 binding of
  the C++ core, §11.4) is in scope but explicitly limited to the
  numerical class `MOC_GridCalc`. The MFC dialogs, plotting widgets,
  and Visual Studio project files are not.

---

*Document owner:* (you). Update each phase's checkboxes as work lands.
Do not delete the "what the other model got wrong" section — it is the
canonical reason future contributors should not re-do this same audit.
