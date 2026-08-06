# Exact Rao/JAX contour architecture for engine MDO

**Status:** design checkpoint only; not implemented  
**Decision owner:** repository maintainer  
**Scope:** replacing the Rao/TOP chart-and-Bézier contour inside the
differentiable engine model without hiding failures or breaking gradients

## 1. Decision

Do **not** call `solve_rao_bvp_jax()` directly from `mdo/grid.py`.

The current public Rao function is an excellent exact-analysis backend, but it
is not yet the function the MDO needs:

- Python/NumPy still owns seed construction, topology, reliability gating, and
  output assembly.
- JAX owns the inner bounded least-squares step, not the complete
  design-parameter-to-wall map.
- The current live-kernel path traces `theta_B`, while the transonic start line
  is still fixed. Full derivatives with respect to `Rt`, `gamma`, and throat
  curvature therefore need one more lift.
- The number and identity of active characteristic rows can change. A derivative
  taken across that topology change is not a derivative of one smooth function.
- `grid.py` needs a fixed-shape wall profile every evaluation; the public Rao
  result is a rich host object with variable-length topology.

The near-term architecture is therefore:

1. retain the analytic TOP contour for inexpensive bulk optimization;
2. make the exact Rao solve a pure-array, explicitly gated analysis;
3. use it for post-optimum re-evaluation and existence checks first; and
4. only then allow a trust-region multifidelity MDO to query it during
   optimization.

This preserves the useful speed of the current MDO while creating a path to
exact Rao sensitivities. It is not a permanent endorsement of chart angles as
the authoritative contour.

**Evidence boundary.** Rao's variational construction and its valid-region
boundary are source-backed. The particular KKT/IFT software architecture below
is an engineering design derived for this repository; the local propulsion
corpus does not contain a primary MDO/IFT methods reference and it must not be
described as corpus-validated.

## 2. Public interfaces

### 2.1 Traced design parameters

```python
class RaoDesignParams(NamedTuple):
    Rt: Array
    epsilon: Array
    length_pct: Array
    gamma: Array
    Pa_over_Pc: Array
    Ru_over_Rt: Array
    Rd_over_Rt: Array
```

All seven values are JAX leaves. `Pc` and `Pa` may be carried separately by the
engine state, but the contour/separation layer only needs their ratio.

### 2.2 Static topology

```python
class RaoStaticTopology(NamedTuple):
    n_start: int
    n_kernel_rows_max: int
    n_kernel_width_max: int
    n_control: int
    n_wall: int
    n_wall_sample: int
    residual_layout: ...
    masks_and_connectivity: ...
```

This object contains only compile-time sizes, integer connectivity, and masks.
It is created host-side for a continuation window and passed as a genuinely
static JIT argument (or captured by the compiled closure), never as a traced
array leaf. All variable-width characteristic rows are padded with finite
sentinel states and accompanied by an explicit validity mask.

### 2.3 Pure numerical state

```python
class RaoState(NamedTuple):
    primal_unknowns: Array
    equality_multipliers: Array
    start_line: FlowField
    kernel_rows: FlowField
    kernel_mask: Array
    control_surface: FlowField
    wall_nodes: FlowField
    wall_sample_x: Array
    wall_sample_r: Array
    wall_sample_s: Array
    residual_vector: Array
    kkt_residual_vector: Array
    residual_block_norms: Array
    constraint_vector: Array
    validity_vector: Array
```

There are no strings, Python dictionaries, variable-length lists, or host
objects in this state. Host-side provenance and human-readable failure reasons
are added by the analysis snapshot.

## 3. Unknown-vector layout

For `n_c` control-surface nodes and `n_w` wall nodes, preserve the existing
layout and make `theta_B` explicit:

```text
M_ce[n_c], theta_ce[n_c], r_ce[n_c],
M_wall[n_w], theta_wall[n_w], x_wall[n_w], r_wall[n_w],
lambda_2, lambda_3, log_C, k_D_fraction,
pair_fraction[n_c],
theta_B
```

The total is `4*n_c + 4*n_w + 5` **primal** unknowns when the wall is coupled.
The square solve appends one equality multiplier per hard mass/length/endpoint
constraint; both the primal and multiplier arrays are explicit in `RaoState`.
This is compatible with the existing JAX assembly and live-`theta_B` kernel
march, so the migration does not require a second Rao formulation.

The transonic start-line state must become a traced function of
`(Rt, gamma, Ru/Rt, Rd/Rt)`. Until that lift is complete, derivatives in those
directions must be marked unavailable; freezing the start line and reporting
the result as a total derivative would be incorrect.

## 4. One pure JAX problem definition

The implementation boundary should be:

```python
rao_problem(params, topology, seed) -> (kkt_residual_fn, initial_state)
solve_rao_state(params, topology, seed) -> RaoState
```

One evaluation of the problem must recompute, in graph:

- the transonic start line;
- the padded throat-arc/kernel march;
- the B–D interpolation and partial mass flux;
- the D–E control surface;
- the coupled wall and characteristic pairing;
- mass and prescribed-length closure; and
- exit and wall endpoint conditions.

The existing residual families remain individually visible:

```text
mass
length
Rao stationarity
C+ compatibility
C- compatibility
D-state continuity
control-surface geometry
wall endpoints
wall tangency
CE-to-wall C+ compatibility
CE-to-wall intersection
```

Regularization is never allowed to make a failed physics block appear
converged. Physics residuals and numerical regularizers are reported and gated
separately.

## 5. Square solve and implicit derivatives

The present overdetermined least-squares residual is useful for finding a
solution, but the differentiable contract should be a square system. Use the
KKT equations of a constrained residual problem:

```text
minimize_z    1/2 ||r_physics(z, p)||_W^2 + 1/2 ||r_regularization(z, p)||^2
subject to    c_mass,length,endpoints(z, p) = 0
```

With equality multipliers `lambda_c`, solve

```text
F(z, lambda_c, p) =
[
  J_r(z,p)^T W r(z,p) + J_reg(z,p)^T r_reg(z,p)
  + J_c(z,p)^T lambda_c
  c(z,p)
] = 0.
```

This is square in `(z, lambda_c)`. A damped Newton or trust-region root solve
uses warm starts and a continuation ladder. The returned design derivative is
the implicit derivative

```text
d(z,lambda_c)/dp = -F_(z,lambda_c)^(-1) F_p.
```

For the engine MDO's small design vector, direct/forward solves are appropriate.
An adjoint solve can be added for scalar objectives without changing the primal
contract.

The KKT stationarity condition is **not** itself evidence that the Rao physics
closed. Acceptance additionally requires every named unweighted physics
residual to meet its own tolerance.

## 6. Bounds and inequalities

Sigmoid reparameterization can remain for genuinely bounded scalar unknowns,
but physical validity is represented explicitly, not through hidden clipping.

The following become returned inequalities or hard gates:

- positive radius and supersonic Mach where required;
- `0 <= k_D_fraction <= 1`;
- monotone pairing fractions;
- non-negative axial advance of the manufactured wall;
- positive boundary-function/radial-increment validity along D–E;
- no exhausted kernel padding;
- `theta_B` inside the current smooth topology window;
- finite state and residual arrays;
- residual, mass, length, attachment, and exit-landing tolerances;
- acceptable linear-system conditioning; and
- requested separation margin at the operating pressure ratio.

If an iterate crosses a topology window, the host continuation driver rebuilds
the topology and restarts from the last accepted state. It does not silently
differentiate across the change.

## 7. Fixed-shape wall handoff

Cooling must not depend on a variable number of Rao nodes. After an accepted
solve:

1. join the solved throat arc and solved B–E wall;
2. compute normalized wall arc length `s_hat` in `[0, 1]`;
3. interpolate `x`, `r`, and the flow state onto a fixed
   `n_wall_sample` grid;
4. preserve a throat index and region masks; and
5. rebuild `area_ratio`, Mach, and segment lengths consistently.

The sampler is pure JAX. CAD and reports receive the full host contour after
optimization; the fixed sample is the authoritative numerical profile used by
the MDO and parity tests.

## 8. Continuation and initialization

The host driver owns a cache keyed by:

```text
(gamma window, epsilon window, length_pct window,
 Ru/Rt, Rd/Rt, resolution, residual-layout version)
```

Initialization proceeds in this order:

1. analytic TOP contour supplies wall and angle seeds;
2. the existing Rao construction supplies a physically valid reference seed;
3. nearby designs use pseudo-arclength or parameter continuation;
4. each accepted solution becomes the next warm start; and
5. a topology rebuild occurs before padding or angle-window exhaustion.

A cold solve that cannot meet the gates returns an explicit invalid state. It
never substitutes the TOP wall under an `"exact_rao"` provenance label.

## 9. MDO integration stages

### Stage A — post-optimum exact analysis

Optimize with the current differentiable TOP model. At the optimum, solve the
exact Rao problem, resample its wall, rerun cooling on that wall, rerun the
traditional nozzle and electric-pump analyses, and compare all common scalars
and normalized profiles.  Until non-Bezier CAD clears its own validation gate,
the exact contour produces a preliminary numerical/report artifact only; the
existing Bezier CAD package remains a separate comparison artifact and must not
be relabeled as CAD generated from the exact wall.

### Stage B — existence-aware screening

During optimization, query a cheap differentiable surrogate of exact-Rao
existence and conditioning. Keep the exact solve outside the objective. Reject
designs whose post-optimum exact solve fails.

### Stage C — trust-region multifidelity MDO

Only after primal parity, derivative parity, mesh convergence, and failure-mode
tests pass, allow exact-Rao corrections inside a bounded trust region. Refresh
the correction and derivatives at accepted major iterates. Shrink or reject the
step when the exact model and TOP model disagree beyond their declared error
budget.

### Stage D — exact-Rao-in-the-loop research mode

Permit every evaluation to call the implicit Rao state only when continuation
coverage and runtime make that practical. Keep the TOP path available as an
ablation, not as a silent fallback.

## 10. Acceptance tests

Implementation is not complete until all of these pass:

1. NumPy/JAX parity for every residual block on identical states.
2. Start-line and kernel parity while varying `Rt`, `gamma`, and both throat
   radii.
3. Primal contour parity at the Rao reference and a grid of
   `(epsilon, length_pct)`.
4. `jacfwd`, `jacrev`, and centered finite-difference agreement away from
   topology boundaries.
5. Explicit derivative-unavailable status at a topology boundary.
6. Residual, mass, length, seam, endpoint, monotonicity, and valid-region
   failure injection.
7. Fixed-wall-sampler conservation and normalized-profile parity.
8. Resolution refinement of contour, thrust coefficient, heat flux, wall
   temperature, and reduced-Hessian quantities.
9. Warm-start and cold-start branch consistency.
10. End-to-end post-optimum parity through cooling, performance, reports, and
    CAD.

## 11. Failure semantics

The exact solver returns three separate judgments:

- **numerical convergence:** the square solve terminated with finite arrays;
- **physics closure:** every named unweighted residual met tolerance; and
- **Rao validity:** the characteristic construction remained in the valid
  region with unused padding and acceptable conditioning.

Only all three together make the exact wall available. The host snapshot records
the failed judgment, residual norms, continuation window, topology version, and
seed provenance. Unsupported or invalid downstream quantities are `None` with
an availability reason.

## 12. Approval checkpoint

No exact-Rao replacement in `mdo/grid.py` should be implemented until this
architecture is accepted or revised. The output-contract and parity work can
land independently because it gives both contour paths a common, auditable
handoff.

## 13. Source record

Direct source claims above were located through
`propulsion_texts/propulsion_texts_for_agents/paper_index.md`, checked in the
Markdown mirror, and verified against the original PDFs:

- G. V. R. Rao, “Exhaust Nozzle Contour for Optimum Thrust” (1958),
  DOI 10.2514/8.7324, sections “Solution of the Problem” and “Method of
  Constructing Optimum Nozzle Contour,” original PDF pp. 3–4 (printed
  pp. 379–380). The source makes D–E a left characteristic, applies
  characteristic compatibility, closes mass by trial selection of D, and
  treats a limiting line as physically inadmissible. Mirror:
  `propulsion_texts/propulsion_texts_for_agents/markdown/rao1958.md`;
  original: `propulsion_texts/rao1958.pdf`.
- G. V. R. Rao, J. Beck, and T. Booth, “Nozzle Optimization for Space-Based
  Vehicles” (1999), AIAA 99-2584, original PDF pp. 2–6, especially “Valid and
  Invalid Regions.” The valid construction requires positive `dR`/boundary
  function; below the minimum length a shock-free solution is unavailable for
  the case studied. Its numeric map is specific to the paper's equilibrium
  O2/H2 cases and is not imported as a universal domain. Mirror:
  `propulsion_texts/propulsion_texts_for_agents/markdown/rao1999.md`;
  original: `propulsion_texts/rao1999.pdf`.
- J. C. Hyde and G. S. Gill, *Liquid Rocket Engine Nozzles* (NASA SP-8120,
  1976), §3.1.2.1.3, original PDF p. 82 (printed p. 68). The “within
  20 percent” separation guidance concerns overexpanded/ground-test operation
  and is not a universal vacuum margin. Mirror:
  `propulsion_texts/propulsion_texts_for_agents/markdown/19770009165.md`;
  original: `propulsion_texts/19770009165.pdf`.

All three corpus conversions are marked `needs_review`; the cited page content
was therefore checked in the original PDFs. The square KKT system, topology
window, fixed-shape sampler, continuation cache, and staged multifidelity policy
are repository design decisions, not quotations or claims attributed to those
papers.
