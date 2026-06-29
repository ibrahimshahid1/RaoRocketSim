# RaoRocketSim

RaoRocketSim is a Python research and preliminary-design toolbox for
axisymmetric rocket-nozzle aerodynamics. It pairs a practical Rao
thrust-optimized parabolic (TOP) bell workflow with quasi-one-dimensional
performance estimation, low-order engineering screens, an axisymmetric
method-of-characteristics (MOC) solver, a finite-dimensional Rao
variational boundary-value problem (BVP), a vendored NASA/JHU
reference-code port, and a differentiable JAX backend with exact-Jacobian
solves and gradient sensitivities.

The **default, trusted** path is the chart-based Rao/TOP quadratic-Bézier
contour: it is deterministic, endpoint-exact, and benchmarked against
published geometry. The MOC and variational solvers are **active research
implementations** — they expose residuals, topology, reference comparisons,
and explicit reliability metadata, but they are not promoted to
design-validated or hardware-qualified status.

> **Engineering status.** Every generated contour is stamped
> `hardware_qualified = False`. Passing the repository's design gates means a
> result cleared internal preliminary screening. It does **not** replace
> independent CFD, conjugate heat-transfer analysis, structural FEA,
> combustion-stability work, material allowables, manufacturing review,
> inspection, proof testing, or hot-fire qualification.

All symbols, ratios, and governing equations in this document have been
cross-checked against the source code and against the primary literature
stored in [`propulsion_texts/`](propulsion_texts/); citations are given
inline and collected at the end.

---

## Table of contents

1. [Capabilities and maturity](#capabilities-and-maturity)
2. [Test and validation status](#test-and-validation-status)
3. [What the tool does](#what-the-tool-does)
4. [Physical model](#physical-model)
5. [Contour methods](#contour-methods)
6. [Differentiable (JAX) backend](#differentiable-jax-backend)
7. [Screening models](#screening-models)
8. [Installation](#installation)
9. [Command-line usage](#command-line-usage)
10. [Python API](#python-api)
11. [Outputs](#outputs)
12. [Validation and reference evidence](#validation-and-reference-evidence)
13. [Repository layout](#repository-layout)
14. [Known limitations](#known-limitations)
15. [Remaining work](#remaining-work)
16. [References](#references)

---

## Capabilities and maturity

| Area | Current implementation | Status |
|---|---|---|
| Rao/TOP geometry | Upstream + downstream throat arcs and a quadratic Bézier bell using interpolated Rao-chart angles | **Trusted preliminary baseline** |
| Ideal performance | Constant-$\gamma$, calorically perfect, quasi-1D isentropic flow; $C_F$, thrust, $I_{sp}$, $c^*$, $\dot m$, exit state | Preliminary |
| Thermochemistry | Built-in combustion-product constants, or optional RocketCEA chamber $\gamma$, $M_w$, $T_c$, $c^*$ | Preliminary; nozzle flow stays constant-$\gamma$ |
| Direct MOC wall optimization | Axisymmetric characteristic march coupled to a monotone spline wall under SLSQP / Nelder–Mead | Experimental |
| Rao variational / MOC BVP | NASA-topology seed, Rao stationarity, characteristic compatibility, mass + length closure, $D$-state continuity, validity and topology diagnostics | Experimental research path |
| Differentiable solver | JAX/Optimistix Levenberg–Marquardt residual solve, differentiable NASA-style kernel march, optional solved $\theta_B$, exact $C_F$ sensitivities | Implemented, research-grade |
| Wall pressure & separation | Quasi-1D wall-pressure estimate; Summerfield, Kalt–Badal, Schmucker separation screens | Screening only |
| Thermal / cooling / structure | Bartz convection, Sieder-Tate rectangular channels with a circular-duct laminar proxy, fin correction, rough/helical Darcy loss, coolant-chemistry screens, station-wise SP-125 liner stress | Screening / preliminary sizing |
| Geometry export | CSV/STL, neutral STEP, Inventor manifest, and optional full-N one-solid regen B-rep with patterned ribs plus plenum/port voids | Preliminary manufacturing geometry, not production definition |
| Validation | Unit/regression tests, literature manifests, NASA/JHU parsers, kernel/topology parity, diagnostic reports | Strong software verification; incomplete physical validation |

The maturity column is enforced in code. Each contour carries a
`design_status` keyed by method —
`preliminary_top_geometry`, `experimental_moc_geometry`,
`experimental_variational_geometry`, or
`experimental_rao_variational_moc_bvp` — and a research-grade
`ContourReliability` level (see
[Reliability labels](#reliability-labels)).

---

## Test and validation status

The most recently recorded run of the normal (non-`slow`) selection, on
**June 14, 2026**, reports:

```text
784 passed, 4 xfailed, 26 deselected in 350.73 s
```

produced with

```bash
MPLCONFIGDIR=/tmp/raosim-mpl \
  .venv-jax/bin/python -m pytest -q -m "not slow"
```

The repository contains 339 test functions, expanded by parametrization to
the counts above. The four expected `xfail`s record known research gaps:
the unresolved provenance of one historical NASA `TT'` fixture, an
unpackaged Cuffel–Back–Massier dataset, and literature-promotion tests for
the experimental MOC and legacy variational paths. The 26 deselected tests
are long JAX solves, convergence studies, NASA fixed-end closure, and the
full Rao-chart sweep, all gated behind the `slow` marker.

This is **software and mathematical verification**. It demonstrates internal
consistency, parity between backends, and agreement with reference fixtures
— not physical validation against experiment or CFD. See
[Validation and reference evidence](#validation-and-reference-evidence).

---

## What the tool does

Given a chamber pressure, ambient pressure, throat size or target thrust,
expansion ratio, propellant model, and contour method, RaoRocketSim can:

- generate a reduced-length Rao/TOP bell, a conical nozzle, an
  MOC-optimized bell, or an experimental Rao variational/MOC contour;
- size throat radius from a requested thrust, or size expansion ratio for
  matched ideal expansion;
- compute ideal exit Mach number and pressure, thrust coefficient, thrust,
  mass flow, effective exhaust velocity, and specific impulse;
- estimate wall pressure, overexpansion separation, altitude performance,
  boundary-layer displacement, heat flux, regenerative-cooling capacity, and
  thin-wall pressure stress;
- generate one injector-to-exit thrust-chamber contour from $L^*$,
  contraction ratio, shoulder geometry, and a shared throat specification;
- sweep $\varepsilon$, $P_c$, or $R_t$; compare contour families; and run
  literature-backed benchmark cases with explicit pass / report / xfail
  policies;
- plot contours, characteristic nets, Mach / pressure / angle fields, wall
  distributions, exit-plane profiles, topology, residual diagnostics, and
  JAX sensitivity fields;
- export versioned CSV, STL, STEP, JSON, Markdown, and metadata artifacts;
- compute exact derivatives of the control-surface $C_F$ with respect to the
  solved node variables via the JAX backend.

---

## Physical model

### Assumptions

The core gas-dynamics and nozzle solvers assume:

- steady, inviscid, adiabatic flow;
- a calorically perfect ideal gas with constant $\gamma$;
- isentropic expansion, except where an empirical loss or separation screen
  is applied;
- axisymmetry for the active MOC and Rao implementations;
- a choked throat and a fully supersonic divergent section;
- **no** reacting-flow chemistry evolution, finite-rate kinetics, particles,
  film cooling, wall roughness, ablation, embedded shocks, side loads, or
  fluid–structure interaction inside the solved flowfield.

The optional CEA integration supplies a chamber-property *snapshot*. The
`cea_frozen` and `cea_equilibrium` modes preserve provenance and
configuration intent, but the nozzle flow is still evaluated with a single
effective chamber $\gamma$; variable-property MOC is not implemented.

### Propellant / thermochemistry table

The built-in database stores nominal **combustion-product** properties (the
exhaust gas at the nominal mixture ratio), not raw propellant properties:

| Propellant | $\gamma$ | $M_w$ [kg/mol] | $T_c$ [K] | $\eta_{Isp}$ | O/F |
|---|---:|---:|---:|---:|---:|
| N2O/Ethanol | 1.22 | 0.0260 | 2800 | 0.92 | 5.5 |
| LOX/RP-1 | 1.23 | 0.0235 | 3400 | 0.96 | 2.6 |
| LOX/LCH4 | 1.24 | 0.0220 | 3500 | 0.96 | 3.5 |
| LOX/LH2 | 1.20 | 0.0100 | 3250 | 0.98 | 6.0 |

The specific gas constant is $R = R_u / M_w$ with
$R_u = 8314.46~\mathrm{J\,kmol^{-1}K^{-1}}$. Users may also supply custom
$\gamma$, $M_w$, $T_c$, and efficiency values, or request RocketCEA-derived
chamber properties.

### Isentropic gas dynamics

For Mach number $M$ and ratio of specific heats $\gamma$, the implemented
stagnation relations are

$$
\frac{T}{T_0}=\left(1+\frac{\gamma-1}{2}M^2\right)^{-1},
\qquad
\frac{p}{p_0}=\left(\frac{T}{T_0}\right)^{\gamma/(\gamma-1)},
\qquad
\frac{\rho}{\rho_0}=\left(\frac{T}{T_0}\right)^{1/(\gamma-1)} .
$$

The area–Mach relation is

$$
\frac{A}{A^*}=\frac{1}{M}
\left[\frac{2}{\gamma+1}\left(1+\frac{\gamma-1}{2}M^2\right)\right]^{\frac{\gamma+1}{2(\gamma-1)}} ,
$$

inverted with Newton iteration on the chosen (subsonic or supersonic)
branch. The Prandtl–Meyer function and Mach angle are

$$
\nu(M)=\sqrt{\frac{\gamma+1}{\gamma-1}}\,
\tan^{-1}\!\sqrt{\frac{\gamma-1}{\gamma+1}\left(M^2-1\right)}
-\tan^{-1}\!\sqrt{M^2-1},
\qquad
\mu=\sin^{-1}\!\left(\frac{1}{M}\right) ,
$$

with $\nu(M)$ inverted by Newton iteration using the analytic derivative
$\mathrm{d}\nu/\mathrm{d}M=\sqrt{M^2-1}\,/\,[M(1+\tfrac{\gamma-1}{2}M^2)]$.
These relations are standard compressible-flow results (Anderson, *Modern
Compressible Flow*; cross-checked against `propulsion_texts/prmeyer.pdf`).

### Performance model

With $\varepsilon = A_e/A_t$, $A_t=\pi R_t^2$, and $A_e=\varepsilon A_t$, the
ideal one-dimensional thrust coefficient is

$$
C_F=
\sqrt{\frac{2\gamma^2}{\gamma-1}
\left(\frac{2}{\gamma+1}\right)^{\frac{\gamma+1}{\gamma-1}}
\left[1-\left(\frac{p_e}{p_c}\right)^{\frac{\gamma-1}{\gamma}}\right]}
+\left(\frac{p_e-p_a}{p_c}\right)\varepsilon .
$$

The built-in propellant model applies an empirical efficiency multiplier
$\eta_{Isp}$ to the complete ideal coefficient:

$$
C_{F,\mathrm{actual}}=\eta_{Isp}\,C_F,
\qquad
F=C_{F,\mathrm{actual}}\,p_c A_t .
$$

The characteristic velocity, mass flow, specific impulse, and effective
exhaust velocity are

$$
c^*=\frac{\sqrt{\gamma R T_c}}{\gamma\sqrt{\left(2/(\gamma+1)\right)^{(\gamma+1)/(\gamma-1)}}},
\qquad
\dot m=\frac{p_c A_t}{c^*},
\qquad
I_{sp}=\frac{C_{F,\mathrm{actual}}\,c^*}{g_0},
\qquad
V_e=I_{sp}\,g_0 ,
$$

with $g_0 = 9.80665~\mathrm{m\,s^{-2}}$. These are **ideal-cycle** estimates.
The efficiency multiplier is a single lumped factor, not a resolved loss
model, and must not be read as a prediction of combustion, boundary-layer,
two-phase, or chemical losses. (Formulation: Sutton & Biblarz, *Rocket
Propulsion Elements*; Anderson.)

---

## Contour methods

### Conical reference

The conical utility reuses the bell throat arcs and a straight divergent
wall at half-angle $\alpha$. Its length and classical divergence factor are

$$
L_{\mathrm{cone}}=\frac{R_e-R_t}{\tan\alpha},
\qquad
\eta_{\mathrm{div}}=\frac{1+\cos\alpha}{2} .
$$

The comparison module also estimates a bell divergence loss from an assumed
linear exit-plane flow-angle profile,

$$
\eta_{\mathrm{div}}\approx
\frac{\displaystyle\int_0^{R_e}\rho u_x\,|\mathbf{u}|\,r\,\mathrm{d}r}
{\displaystyle\int_0^{R_e}\rho\,|\mathbf{u}|^2\,r\,\mathrm{d}r},
\qquad
C_{F,2D}\approx\eta_{\mathrm{div}}\,C_{F,1D} ,
$$

which is a comparison aid, **not** a resolved exit-plane solution of the
Bézier contour.

### 1. Rao/TOP Bézier baseline — `method="bezier"` (default, trusted)

The baseline contour has three pieces:

1. an upstream circular throat arc, $R_u = 1.5\,R_t$;
2. a downstream circular throat arc, $R_d = 0.382\,R_t$;
3. a quadratic Bézier bell from inflection point $N$ to exit point $E$.

The exit radius and reference 15° cone length are

$$
R_e=R_t\sqrt{\varepsilon},
\qquad
L_{15}=\frac{R_e-R_t}{\tan 15^\circ},
\qquad
L_n=\frac{L_{\%}}{100}\,L_{15} ,
$$

and the bell is

$$
\mathbf{B}(t)=(1-t)^2\,\mathbf{N}+2(1-t)t\,\mathbf{P}_1+t^2\,\mathbf{E},
\qquad 0\le t\le 1 ,
$$

where $\mathbf{P}_1$ is the intersection of the tangent leaving $N$ at angle
$\theta_n$ and the tangent entering $E$ at angle $\theta_e$.

**Chart angles and provenance.** The default $(\theta_n,\theta_e)$ are
bilinearly interpolated from embedded Rao/NASA chart tables spanning
approximately $4\le\varepsilon\le 50$ and $60\%\le L_{\%}\le 100\%$. The
tables reproduce Rao's TOP design charts (G. V. R. Rao, *Approximation of
Optimum Thrust Nozzle Contour*, ARS J. 30(6), 1960; as reproduced in Sutton
and NASA SP-8120). Rao's underlying optimum study was computed for
$\gamma = 1.23$; per Rao, *Recent Developments in Rocket Nozzle
Configurations* (ARS J. 31(11), 1961), the optimal **contour** is nearly
$\gamma$-insensitive at fixed $(\varepsilon, L)$ — only $C_F$ depends
strongly on $\gamma$ — so the angle tables remain valid comparison targets
at other $\gamma$. Inputs outside the grid are linearly extrapolated by the
interpolator and should be treated cautiously.

This method is the trusted preliminary baseline because it is deterministic,
smooth, endpoint-exact, benchmarked against explicit TOP geometry, and does
not depend on the convergence of an experimental flow solver.

### Chamber and convergent geometry

The authoritative contour includes the injector face, cylindrical chamber,
rounded chamber shoulder, straight convergent, shared upstream throat arc,
downstream throat arc, and bell. Chamber sizing uses contraction ratio
$CR=A_c/A_t$ and characteristic length $L^*=V_c/A_t$:

$$
R_c=R_t\sqrt{CR},
\qquad
V_c=L^* A_t .
$$

The cylindrical length is root-solved against the exact volume of the
piecewise-linear revolved contour. Each meridional segment is integrated as a
conical frustum, including the shoulder and upstream throat arc. Infeasible
combinations are rejected rather than assigned an arbitrary short cylinder.

`ThroatGeometrySpec` is shared by chamber and nozzle and owns $R_u/R_t$,
$R_d/R_t$, convergent angle, and throat location. The minimum cylindrical
length and chamber-shoulder radius factor are explicit design inputs; their
fallback values are geometric placeholders, not injector- or
combustion-qualified limits.

This remains a **geometric volume model only**: $L^*$ is a residence-time
proxy, and its minimum useful value depends on propellants, injector/mixing,
pressure, mixture ratio, packaging, stability, cooling, and hot-fire evidence.

### 2. Direct MOC wall optimization — `method="moc"`

This path parameterizes the bell as a monotone cubic Hermite spline with
decision variables

$$
\mathbf{q}=\left[\theta_n,\,r_1,\,r_2,\dots,r_{n_c}\right] .
$$

For each candidate wall it builds a transonic starting line, marches an
axisymmetric characteristic net with wall feedback, samples the exit plane,
and minimizes a cost combining negative exit thrust, exit-flow-angle
penalties, radius monotonicity, and curvature regularization. SciPy SLSQP is
used when available, with a NumPy Nelder–Mead fallback. The exported bell is
currently *reconstructed* as a smooth quadratic Bézier from the optimized
entrance and exit angles rather than exporting the sparse spline directly —
one reason the method remains `experimental_moc_geometry`.

### 3. Legacy direct variational path — `method="rao"`

The legacy Rao path discretizes a supersonic control surface, evaluates
thrust / mass-flow / length functionals, solves a finite-dimensional
constrained optimization, and attempts a control-surface-driven MOC wall
construction. It is retained for regression and comparison; its public
status is `experimental_variational_geometry`, and literature-promotion
tests are expected to fail.

### 4. Rao variational / MOC BVP — `method="rao_variational_moc"` (main research solver)

This solver seeds a finite-dimensional BVP from the NASA/JHU topology

$$
T T' \;\rightarrow\; B \;\rightarrow\; BD \;\rightarrow\; D \;\rightarrow\; DE \;\rightarrow\; E .
$$

Its default configuration uses the JAX/Optimistix Levenberg–Marquardt
backend with exact autodiff Jacobians, the characteristic residual
formulation, a NASA-style fixed-end topology seed, full position /
flow-angle / Mach continuity at point $D$, a continuation ladder that raises
weights on the mass, length, and endpoint constraints, and separate raw-wall,
export-wall, residual, topology, and reliability diagnostics. The solved
control-surface unknowns are, schematically,

$$
\mathbf{u}=\left[\,M_i,\;\theta_i,\;r_i \;\middle|\;
\lambda_2,\;\lambda_3,\;\log C,\;f_D\,\right],
$$

with optional wall unknowns and an optional live kernel angle $\theta_B$.
Here $f_D$ is the arc-length fraction locating $D$ on the kernel
characteristic $BD$.

#### Axisymmetric characteristics

With nodes ordered downstream and the axisymmetric source

$$
S=\frac{\sin\theta\,\sin\mu}{r},
$$

the implemented compatibility relations are

$$
C^{+}:\quad \frac{\mathrm{d}r}{\mathrm{d}x}=\tan(\theta+\mu),
\qquad \mathrm{d}(\theta-\nu)=-S\,\mathrm{d}s,
$$

$$
C^{-}:\quad \frac{\mathrm{d}r}{\mathrm{d}x}=\tan(\theta-\mu),
\qquad \mathrm{d}(\theta+\nu)=+S\,\mathrm{d}s .
$$

The $C^{+}$ control surface $DE$ is enforced geometrically by reconstructing
its axial coordinates,

$$
x_{i+1}=x_i+\frac{r_{i+1}-r_i}{\tan(\bar\theta_i+\bar\mu_i)},
\qquad x_1=x_D(f_D) .
$$

The source term $S=\sin\theta\sin\mu/r$ and the $C^{\pm}$ invariant pairing
are taken from Anderson (*Modern Compressible Flow* §11.4) and Zucrow &
Hoffman (*Gas Dynamics* Vol. 2, Ch. 17), and are oracle-validated against the
NASA M3.5Perf grids (the corrected forms hold at RMS $\sim 10^{-6}$–$10^{-8}$
on kernel and Deriv characteristics; see `raosim/rao_residuals.py`).

#### Rao stationarity

The critical Mach number is

$$
M^*=\sqrt{\frac{(\gamma+1)M^2}{2+(\gamma-1)M^2}} .
$$

The primary algebraic optimum-thrust condition along the control surface
$DE$ is

$$
M^*\frac{\cos(\theta-\alpha)}{\cos\alpha}=C,
\qquad \alpha=\mu=\sin^{-1}(1/M),
$$

implemented in logarithmic form for conditioning. This is Eq. (1) of
Rao, Beck & Booth, AIAA 99-2584 (`propulsion_texts/rao1999.pdf`). Its
differential identity, Eq. (2) of the same paper,

$$
\mathrm{d}\ln M^*-(\mathrm{d}\theta-\mathrm{d}\alpha)\tan(\theta-\alpha)
+\mathrm{d}\alpha\tan\alpha=0 ,
$$

is retained as a secondary consistency check. The older direct-method
functionals are also available; per unit radial extent their
stagnation-normalized forms are

$$
f_1=2\pi r\left[\left(\frac{p}{p_0}-\frac{p_a}{p_0}\right)
+\frac{\rho}{\rho_0}\gamma M^2\frac{T}{T_0}
\frac{\sin(\phi-\theta)\cos\theta}{\sin\phi}\right],
$$

$$
f_2=2\pi r\frac{\rho}{\rho_0}\bar V\frac{\sin(\phi-\theta)}{\sin\phi},
\qquad
f_3=\cot\phi ,
$$

representing axial thrust, mass flow, and axial length respectively.

#### Mass and endpoint closure

Mass closure compares the surface-normal flux integral on $DE$ and the
selected $BD$ segment,

$$
\dot m_\Gamma=\int_\Gamma 2\pi r\,\rho V\,\left|\sin(\beta-\theta)\right|\,\mathrm{d}s,
\qquad
\dot m_{DE}=\dot m_{BD} .
$$

The BVP additionally pins the commanded exit station and radius,
$x_E=L_n$ and $r_E=R_t\sqrt{\varepsilon}$, and reports scaled residuals for
mass, length, stationarity, characteristic compatibility, geometry,
regularization, penalties, wall endpoints, and wall tangency.

#### Smooth-flow validity

The classical Rao validity inequality is evaluated along $DE$:

$$
b=1-\frac{\mathrm{d}\alpha}{\mathrm{d}\theta}\,
\frac{\tan(\theta-\alpha)+\tan\alpha}{\tan(\theta-\alpha)-\tan\alpha}\;\ge\;0 .
$$

If $b$ drops below tolerance, the requested smooth variational solution lies
outside the shock-free Rao region; the construction is flagged
`invalid_short_nozzle_region` and the reliability level is downgraded. This
boundary relation is the perfect-gas optimum-thrust boundary of
Rao, Beck & Booth (1999) and is discussed in Östlund's KTH thesis
(`propulsion_texts/fulltext01.pdf`, §3).

For the repository's $\varepsilon=10$, 80%-length, $\gamma=1.4$ regression
case, the smooth stationary-$DE$ reference is

```text
theta_B = 25.5659 deg
f_D     = 0.15216
D       = (M = 3.40145, theta = 18.5182 deg)
E       = (M = 3.47655, theta = 11.1193 deg)
```

These are a numerical regression point for the implemented formulation, not
a universal nozzle-design result.

### Transonic starting lines

The MOC code provides several throat starting-line models:

- `kliegel_levine` — a third-order toroidal-coordinate Kliegel–Levine series
  (default; Kliegel & Levine, 1969);
- `sauer_modified` — a compact leading-order curved-throat approximation;
- `area_ratio` — a quasi-1D area–Mach starting line;
- `hall` — a deprecated alias for `sauer_modified`;
- `nasa_visible_kliegel_levine` — a source-faithful NASA/JHU compatibility
  mode used by the reference port.

The theory-correct Kliegel–Levine implementation documents and tests several
coefficient-transcription corrections; the source-faithful mode deliberately
preserves the historical C++ integer-division behavior required for NASA
binary/output parity. The two modes answer different verification questions
and are intentionally **not** collapsed into one implementation.

---

## Differentiable (JAX) backend

`raosim/jax/` provides the default inner solver for the Rao BVP and the
gradient API. The design keeps a single boundary
(`JAX_DIFFERENTIABLE_PLAN.md` §2): the NumPy shell in
`rao_variational.solve_rao_bvp` still owns seeding, kernel construction,
reliability gating, diagnostics, and output assembly, while the JAX layer
owns only the *inner least-squares solve*.

- **Solver.** Optimistix Levenberg–Marquardt with the **exact** `jacfwd`
  Jacobian of the assembled residual, replacing SciPy's finite-difference
  `least_squares`. Box bounds are handled by a smooth reparametrization
  $u = \mathrm{lo} + (\mathrm{hi}-\mathrm{lo})\,\sigma(z)$ so every iterate
  stays strictly inside the box. A constraint-weight continuation ladder
  up-weights the single-element mass/length/endpoint residuals; the
  *reported* residual is always the unweighted one, so reliability gates stay
  comparable across the NumPy and JAX backends.
- **Differentiable kernel march and $\theta_B$.** The NASA-style kernel march
  and the live $\theta_B$ secant path are differentiable.
- **Sensitivities.** `rao_sensitivities(config, solution=...)` returns exact
  derivatives of the control-surface $C_F$ with respect to the solved node
  variables, explicit fixed-solution partials with respect to $p_a/p_0$ and
  $\gamma$, and the residual-Jacobian condition number.

**Not yet provided (v2 deferrals):** total re-solved (implicit-function)
design derivatives with respect to $R_t$, $\varepsilon$, length percentage,
or $\gamma$; a differentiable sensitivity map carried through to the final
manufacturable bell wall; and the Hessian. The pinned, tested stack is
`jax==0.6.2`, `jaxlib==0.6.2`, `optimistix==0.0.11`, `equinox==0.13.8`.

---

## Screening models

These models are deliberately low-order. They are useful for ranking
concepts and rejecting obviously unsuitable designs, **not** for
qualification.

### Wall pressure

At each contour radius the code sets
$A(x)/A_t=\left(r(x)/R_t\right)^2$, inverts the area–Mach relation, and
evaluates $p(x)/p_c$ isentropically. A positive downstream pressure
increment is flagged as non-monotonic (a check motivated by NASA SP-8120).
This is a quasi-1D estimate, not a boundary-layer or shock solution.

### Separation

Three empirical overexpansion-separation criteria are exposed (as
**implemented**):

$$
p_{\mathrm{sep}}\approx 0.4\,p_a \quad\text{(Summerfield)},
$$

$$
\frac{p_{\mathrm{sep}}}{p_a}\approx\frac{1}{1.88\,M_e-1}\quad\text{(Kalt–Badal implementation)},
$$

$$
\frac{p_{\mathrm{sep}}}{p_c}\approx\left(\frac{p_a}{p_c}\right)^{0.8}M_e^{-1}\quad\text{(Schmucker implementation)} .
$$

The first contour station whose quasi-1D wall pressure falls below the chosen
threshold is reported as the estimated separation location. Side loads,
restricted/free shock separation, hysteresis, and transient startup are not
modeled. (Criteria surveyed in NASA SP-8120 and Stark, *Flow Separation in
Rocket Nozzles — An Overview*, 2009.)

### Boundary layer

The displacement-thickness screen uses a turbulent flat-plate-style
correlation,

$$
\delta^*\approx 0.046\,\frac{s}{Re_s^{1/5}}\sqrt{\frac{T}{T_w}},
$$

and estimates an effective exit area ratio from $r_e-\delta_e^*$. Gas
viscosity uses a Sutherland-type estimate.

### Heat flux and cooling

The gas-side model implements the Bartz correlation, including the pressure,
area-ratio, throat-curvature, and wall-temperature property factor. Gas
transport properties remain estimated unless supplied from CEA or measured
data, so it is still preliminary design physics rather than a validated CHT
boundary condition. Regenerative cooling is marched station by station in
the coolant-flow direction. The coolant-side coefficient uses Sieder–Tate
with local rectangular hydraulic diameter, bulk/wall viscosity, a `Nu=4.36`
circular-duct laminar proxy, and fin area. Rectangular laminar aspect-ratio,
heated-wall, and developing-flow effects remain unresolved. The optional
curved-channel multiplier is the Niino-Kumakawa/Taylor relation reproduced by
Pizzarelli et al. (2011) and Torres et al. (2009). It remains disabled in
automatic liquid-coolant sizing because NASA SP-8087 recommends experimental
calibration before crediting curvature enhancement;
`--curvature-correction` opts into it. The wall is the series
resistance

$$
\frac{1}{H(x)}=\frac{1}{h_g(x)}+\frac{t_\mathrm{hot}(x)}{k_w}
               +\frac{1}{h_c(x)} .
$$

Total absorbed heat and coolant rise satisfy

$$
\dot Q=\int q''(s)\,2\pi r(s)\,\mathrm{d}s,
\qquad
\Delta T_c=\frac{\dot Q}{\dot m_c\,c_{p,c}},
$$

while friction pressure loss is integrated with local Darcy factor,
hydraulic diameter, velocity, channel roughness, and the actual helical
passage length. Smooth passages use Blasius and rough passages use
Swamee-Jain. An
absolute coolant outlet pressure (or the explicit default
$p_{c,\mathrm{out}}=P_c+\Delta p_\mathrm{injector}$) anchors the jacket
pressure march. RP-1/kerosene reports a coolant-side wall-temperature margin
against a conservative 700 K coking screen and can gate sizing with
`--gate-coolant-chemistry`. `--hydraulic-network` adds every channel branch,
annular inlet/outlet headers, ports, and entry/exit losses to the pressure
budget. Methane and LH2 use station-wise CoolProp properties with
`--coolant-property-backend auto|coolprop`. `--radiation-model spectral`
accepts explicit band data, while `leccese_gray` provides a documented
LOX/CH4 or LOX/H2 screen. `--boiling-chf` reports phase state and a
conservative Zuber CHF reference; supercritical heat-transfer deterioration
and qualified forced-flow cryogenic CHF remain outside the model.

### Structural screen

For a coaxial-shell regen liner, the station-wise combined stress follows
NASA SP-125 equation 4-31:

$$
\sigma_c(x)=
\frac{\left[p_\mathrm{cool}(x)-p_g(x)\right]r(x)}{t_\mathrm{hot}(x)}
+
\frac{E\alpha q''(x)t_\mathrm{hot}(x)}
     {2(1-\nu)k_w}.
$$

This exposes the real thickness squeeze: pressure stress wants a thicker
liner, while wall temperature and thermal-gradient stress usually want a
thinner one. `--size-wall` searches hot-wall thickness, channel count, and
channel width together against thermal margin, yield margin, channel fit,
manufacturing minimum, and pressure-drop budget. NARloy-Z uses NASA
CR-134627 Coffin–Manson/Basquin data; GRCop-84 uses NASA direct
total-strain/life regressions. Both are active sourced screening gates, not
chamber-life qualification.

The joint optimizer first selects channels and a feasible uniform
$t_\mathrm{hot}$ candidate, then `size_wall_profile` refines the liner and
outer-jacket thickness and channel depth station by station. SP-125 equation
4-29 is reported as an equivalent-tube longitudinal-buckling screen for
milled channels and does not gate by default; the separate rib-supported
coolant-over-gas liner screen does gate. NARloy-Z and GRCop-84 use sourced
stress- and temperature-dependent cyclic tangent-modulus curves instead of
the former fixed `Et/E` assumption. Fatigue still uses a nominal
$\alpha\Delta T$ strain range rather than a local cyclic-plasticity solution.

Without `--size-wall`, the scalar `--wall-thickness` is only a uniform
geometry/reference input and is labeled as unsized in the report. The
station-wise solver does not force the throat to be thicker: SP-8087 also
recommends thin walls where heat load is highest and nozzle tapering for
heat-flux-limited coolants. Local thermal, pressure, buckling, manufacturing,
and degradation constraints determine the profile; `--t-hot-min` supplies
the process-specific liner floor.

With `--regen-brep`, the CAD path exports one OpenCascade STEP solid
containing liner, full-count patterned ribs, end seals, jacket, and real
channel gaps. `--regen-manifolds` additionally cuts connected annular
plenums and area-sized radial ports. This is a neutral final B-rep, not native
Inventor feature history. The optional one-dimensional manifold graph now
reports maldistribution and local/network losses; 3-D manifold CFD, jacket
buckling, plasticity, creep, weld/braze allowables, stress concentrations,
and nonlinear FEA are not yet solved.

### How user inputs affect wall sizing

- `Rt`, `epsilon`, and `length-pct` set local radius, heated area, passage
  length, channel pitch, and the radius multiplying pressure stress.
- `gamma` and `Pc` change the gas state and Bartz heat loading; `Pc` also sets
  propellant flow and the coolant/gas pressure load.
- `pa/p0` primarily changes nozzle performance/separation. It does not directly
  prescribe thickness.
- `O/F` sets the fuel share used as coolant in the simplified cycle:
  $\dot m_c=\dot m_\mathrm{total}/(1+O/F)$. Lower `O/F` therefore increases
  available fuel coolant in this model; a real change also requires updated
  thermochemistry.
- `margin` and `max wall T` impose the allowable peak wall temperature.
- `pressure-drop budget`, channel width/height/count, coolant properties, and
  helix turns control velocity, heat transfer, and hydraulic loss. Helical
  channels use their longer 3-D path and the reduced pitch transverse to
  coolant flow, not merely a visual coil.
- `auto-size` selects channel geometry at a fixed wall thickness.
  `size-wall` co-sizes channels, then refines variable liner and jacket
  thickness profiles.

The equation provenance, local-PDF inventory, parameter map, and exact CAD
scope are collected in [`docs/regen_wall_model.md`](docs/regen_wall_model.md).

### Atmosphere and trajectory

Altitude performance uses piecewise ISA layers (troposphere −6.5 K/km from
0–11 km, isothermal 11–20 km, +1.0 K/km 20–47 km) with an exponential tail
above 47 km. The optional trajectory module integrates a one-dimensional
vertical point mass,

$$
m\,\dot v=F-\tfrac{1}{2}\rho v|v|\,C_D A-m\,g(h),
\qquad
g(h)=g_0\left(\frac{R_E}{R_E+h}\right)^2,
$$

with $R_E=6.371\times10^{6}~\mathrm{m}$. It is not wired into the main CLI and
does not model guidance, pitch, staging, throttling, winds, or
six-degree-of-freedom dynamics.

---

## Installation

Python 3.12 is recommended because the differentiable backend is pinned to a
tested JAX stack.

For a normal user install after the first PyPI release:

```bash
python3.12 -m pip install RaoRocketSim
```

Optional integrations can be requested with extras:

```bash
# Optional thermochemistry (variable chamber properties)
python3.12 -m pip install "RaoRocketSim[cea]"

# Optional true revolved B-rep STEP export; otherwise a faceted STEP is used
python3.12 -m pip install "RaoRocketSim[cad]"
```

Until the package is published on PyPI, install directly from GitHub:

```bash
python3.12 -m pip install \
  "RaoRocketSim @ git+https://github.com/ibrahimshahid1/RaoRocketSim.git"
```

For repository development:

```bash
python3.12 -m venv .venv-jax
source .venv-jax/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Core dependencies are NumPy, SciPy, Matplotlib, CoolProp, JAX, JAXlib,
Optimistix, and Equinox. Optional integrations are not installed by default.
The PyPI release checklist is in
[`docs/pypi_release.md`](docs/pypi_release.md).

After installation, the toolbox CLI is available as:

```bash
RaoRocketSim --help
RaoRocketSim
```

`RaoRocketSim` launches the current nozzle/chamber/regenerative-cooling/pintle
runner, the same implementation as `python scripts/run_nozzle.py ...`. The
older top-level toolbox interface is still available as:

```bash
RaoRocketSimLegacy --help
```

To ensure a triangle-based STEP fallback is never mistaken for editable
B-rep CAD:

```bash
RaoRocketSim --max-nfev 4000 --regen \
  --l-star 1.0 --contraction-ratio 2.5 \
  --material grcop-84 --size-wall --cad step --require-brep --regen-brep \
  --out builds/regen_brep
```

Add `--regen-manifolds` to include two connected plenums and radial ports.
Port area defaults to total channel flow area and can be distributed with
`--regen-ports-per-manifold`. That flag also enables the matching
full-channel hydraulic graph. The fluid/radiation/phase screens can be run
without CAD, for example:

```bash
RaoRocketSim --max-nfev 4000 --regen --thermal \
  --coolant methane --coolant-inlet-temperature 120 \
  --coolant-property-backend coolprop \
  --hydraulic-network --radiation-model leccese_gray \
  --radiation-family methane --boiling-chf
```

For pintle injector manufacturing geometry, request the machined CAD package:

```bash
RaoRocketSim --injector pintle --injector-cad machined \
  --injector-face-od 0.10 --injector-face-thickness 0.010 \
  --bolt-count 8 --bolt-circle 0.085 --bolt-hole 0.004 \
  --min-tool-diameter 0.0005 --injector-tolerance 0.00005
```

This writes `pintle/injector_manufacturing_report.json` on every run and, when
CadQuery/OpenCascade is installed, exports `injector_face_machined.step`,
`pintle_post_slotted.step`, `annular_sleeve.step`, and
`injector_assembly_machined.step`.

Without `--require-brep`, the summary records either `brep` or
`faceted_brep`. Inventor, Fusion, SolidWorks, and FreeCAD can import the true
STEP as a solid body; it will not contain native Inventor feature history.

For repo-local development, `python scripts/run_nozzle.py ...` remains a thin
compatibility wrapper around the packaged runner.

---

## Command-line usage

CLI pressure units are **bar** for $P_c$, **kPa** for $P_a$, and
**millimeters** for $R_t$ and manufacturing dimensions. Internal APIs use SI
units throughout.

### Preliminary Rao/TOP design

```bash
RaoRocketSim \
  --propellant LOX/RP-1 \
  --Pc 45 --Pa 101.325 \
  --Rt 20 --epsilon 10 --length-pct 80 \
  --method bezier --no-plot \
  --output nozzle_profile.csv
```

Output is written to a versioned directory `builds/vNNN_YYYYMMDD_HHMMSS/`
together with `metadata.txt`.

### Size the throat from thrust

```bash
RaoRocketSim \
  --propellant LOX/LCH4 \
  --Pc 60 --target-thrust 10000 --epsilon 12 --no-plot
```

The sizing relation is

$$
R_t=\sqrt{\frac{F_{\mathrm{target}}}{\pi\,C_{F,\mathrm{actual}}\,p_c}} .
$$

### Experimental Rao variational / MOC solve

```bash
RaoRocketSim \
  --propellant LOX/RP-1 \
  --Pc 45 --Rt 20 --epsilon 10 \
  --method rao_variational_moc \
  --rao-moc-n-control 12 --rao-moc-n-kernel 12 \
  --rao-moc-max-nfev 200 --no-plot
```

This path can be computationally expensive and remains experimental
(`--rao-moc-max-nfev` defaults to 25). `--rao-moc-skip-moc` skips raw
wall/net diagnostics for a faster residual-only study, but that also prevents
MOC reliability promotion.

### Sweep a design variable

```bash
RaoRocketSim \
  --propellant LOX/LCH4 \
  --Pc 60 --Rt 25 --epsilon 10 \
  --sweep epsilon 4 50 20 --no-plot
```

Supported sweep variables are `epsilon`, `Pc`, and `Rt`.

### Run a literature benchmark

```bash
RaoRocketSim \
  --benchmark-case lea_top_schomberg_2014 \
  --benchmark-method bezier \
  --benchmark-report builds/benchmarks --no-plot
```

Available manifests are `lea_top_schomberg_2014`, `rao_scarfed_moc_1990`, and
`vulcain_s1_separation_similarity`. Each metric is classified `strict`,
`xfail`, or `report`, so missing physics is recorded rather than hidden
behind a single pass/fail number.

---

## Python API

### Baseline contour and performance

```python
from raosim.engine import compute_engine_performance
from raosim.nozzle_geometry import bell_nozzle_contour
from raosim.propellants import get_propellant

prop = get_propellant("LOX/RP-1")
contour = bell_nozzle_contour(
    Rt=0.020, epsilon=10.0, length_pct=80.0,
    method="bezier", gamma=prop.gamma,
)
performance = compute_engine_performance(
    Pc=4.5e6, Pa=101325.0, Rt=0.020, epsilon=10.0, prop=prop,
)
```

### Design-gated workflow

```python
from raosim.design import DesignInput, ThermoSpec, design_nozzle_v2

result = design_nozzle_v2(DesignInput(
    thermo=ThermoSpec(mode="constant_gamma", propellant_name="LOX/RP-1"),
    Pc=8.0e6, Rt=0.020, epsilon=8.0,
    method="bezier", mode="preliminary",
))

print(result.performance.thrust)
print(result.gate_report.to_dict())
print(result.report_sections["thermal"])
```

The API name `ValidatedDesignResult` and workflow mode `validated` mean the
stricter internal schema and gate set were used. They do **not** set
`hardware_qualified = True`. Validated mode currently requires RocketCEA, the
Bézier method, regenerative-cooling inputs, material/manufacturing inputs,
and all internal gates to pass.

### Rao BVP and sensitivities

```python
from raosim.rao_variational import RaoSolverConfig, solve_rao_bvp
from raosim.jax import rao_sensitivities

config = RaoSolverConfig(
    Rt=0.020, epsilon=10.0, gamma=1.4, length_pct=80.0,
    solver_backend="jax", formulation="characteristic",
)

solution = solve_rao_bvp(config)
sens = rao_sensitivities(config, solution=solution)

print(solution.reliability)            # ContourReliability level
print(solution.residuals.max_scaled)   # worst scaled residual
print(sens.cf, sens.condition_number)  # Cf and Jacobian conditioning
```

Pass `solver_backend="numpy"` to fall back to the legacy SciPy path.

---

## Outputs

Normal CLI runs create `builds/vNNN_YYYYMMDD_HHMMSS/` and may contain:

- contour CSV in meters;
- a binary STL closed wall solid;
- a STEP solid (CadQuery revolved B-rep when available, else a faceted AP214
  fallback);
- with variable wall sizing, separate liner and closeout-jacket solids;
- optionally, `regen.step`: one full-channel-count cooling-aware material
  solid, with connected plenum/port voids when requested;
- optionally, a pintle machined-CAD package with Boolean-cut STEP bodies for
  the face, slotted pintle post, annular sleeve, and assembly, plus an injector
  manufacturing report;
- an Inventor conversion manifest pointing to the authoritative STEP file;
- optional regen STL/PNG visualization surfaces for liner, channels, and
  jacket;
- design-gate JSON and v2 physics-screening sections;
- sweep CSV or benchmark JSON / Markdown reports;
- human-readable `metadata.txt` with inputs, performance, warnings, gate
  results, and generated filenames.

Native Autodesk Inventor IPT writing is not implemented. Pintle `machined` mode
models injector bolt holes, manifolds, feed/inlet ports, seal grooves, annulus
passages, and slot cut-throughs, but remains preliminary until cold-flow and
joint/fitting-specific validation are complete. Throat inserts, weld/braze
allowances, and non-pintle injector interfaces are metadata/readiness
placeholders, not modeled solid features.

---

## Validation and reference evidence

The repository deliberately separates **software verification** from
**physical validation**.

### Software and mathematical verification

The test suite covers gas-dynamics identities and inverse relations; Bézier,
conical, chamber, curvature, export, and design-workflow behavior;
axisymmetric characteristic pairing and source terms; transonic
Kliegel–Levine coefficients and NASA-visible-source semantics; characteristic
topology, mass conservation, BDE closure, wall tangency, and crossing checks;
NumPy/JAX primitive and residual parity; exact-Jacobian convergence,
kernel-march parity, solved-$\theta_B$ derivatives, and $C_F$ sensitivities;
literature-manifest parsing with strict/report/xfail policies; NASA/JHU
legacy-output and Tecplot parsing; and plotting/diagnostic metadata.

### NASA/JHU reference port

[`Three-Dimensional-Nozzle-Design-Code-master/`](Three-Dimensional-Nozzle-Design-Code-master/)
vendors the Rice/JHU 2D MOC, streamline-tracing, and 3D MOC source package
plus sample outputs. The active Python port implements major pieces of the
axisymmetric 2D workflow: the initial throat line and kernel march; arc-wall,
interior, special-wall, and axis unit processes; mass flow along
right-running characteristics; point-$D$ selection, point-$E$ integration,
and fixed-/free-end closure; the $\theta_B$ secant solution; BDE-region and
wall-contour construction; and explicit `RaoTopology` objects for $TT'$, $B$,
$BF$, $D$, $BD$, $DE$, $E$, and the wall streamline.

The checked-in M3.5 perfect-nozzle wall comparison is within the repository's
$10^{-3}$ RMS regression gate (the perfect-nozzle pipeline currently measures
wall $r(x)$ RMS $\approx 1.8\times10^{-4}$ against `wall.out`). However, the
generator provenance of the historical `TT'.out` fixture is **unresolved**
(see [`docs/nasa_tt_prime_provenance.md`](docs/nasa_tt_prime_provenance.md)):
the visible `MOC_GridCalc_BDE.cpp` source-port workflow is the canonical
reference track, and the orphaned M3.5Perf fixture is a diagnostic overlay
only — **not** promotion authority. Consequently the public solver does not
label any contour `NASA_REFERENCE_MATCHED`.

### Reliability labels

The research code defines an ordered `ContourReliability` vocabulary:

`geometric_approximation` → `moc_compatible` →
`rao_variational_residual_solved` → `nasa_reference_matched` →
`benchmark_validated` → `cfd_checked` → `experimentally_validated`.

This is a *vocabulary*, not a claim that every level has been reached. The
`rao_variational_moc` solver assigns its level from per-solve gates: a clean
solve that passes the BVP residual, MOC closure, valid-region, and thrust
sanity gates reaches `rao_variational_residual_solved`. Promotion to
`benchmark_validated` is gated behind the module flag
`BENCHMARK_VALIDATED_AT_RELEASE`, which is **`False`** in the current code, so
no public contour is benchmark-validated, NASA-reference-matched,
CFD-checked, or experimentally validated by this repository.

---

## Repository layout

```text
main.py                         Primary interactive/batch CLI
raosim/                         Active Python package
  gas_dynamics.py               Perfect-gas, area-Mach, Prandtl-Meyer relations
  engine.py                     Ideal performance model (Cf, Isp, c*, m_dot)
  propellants.py                Combustion-product property database
  nozzle_geometry.py            Rao/TOP Bezier geometry + chart tables
  conical.py                    Conical reference contour and divergence factor
  chamber_geometry.py           L*/CR chamber + convergent sizing
  design.py                     High-level schemas, gates, artifacts
  validation.py                 Design-status labels and engineering gates
  physics.py                    BL, thermal, cooling, structural screens
  separation.py                 Summerfield / Kalt-Badal / Schmucker screens
  wall_pressure.py              Quasi-1D wall-pressure distribution
  atmosphere.py, altitude_performance.py, trajectory.py
                                ISA atmosphere, altitude sweep, 1-D trajectory
  moc.py                        General axisymmetric MOC march + starting lines
  transonic_kernel.py           Kliegel-Levine transonic series
  nasa_moc.py                   NASA/JHU-style kernel and BDE topology port
  moc_topology.py               RaoTopology objects (TT', B, BD, D, DE, E, wall)
  rao_optimizer.py              Direct MOC wall optimization
  rao_variational.py            Rao functionals, validity, and global BVP
  rao_residuals.py              Characteristic residual primitives
  jax/                          Differentiable primitives, residuals, kernel
                                march, LM solve, and sensitivities
  cea.py                        Optional RocketCEA thermochemistry
  benchmarks.py                 Literature benchmark runner
  benchmark_data/               Manifests and digitized reference curves
  plotting.py                   Geometry, field, topology, residual plots
  export.py                     CSV, STL, STEP, IPT-manifest export
tests/                          Unit, regression, parity, and benchmark tests
scripts/                        Research diagnostics and artifact generators
docs/                           NASA provenance, topology, and audit notes
latex-report/                   Mathematical reference report and figures
propulsion_texts/               Local source literature used by the project
Three-Dimensional-Nozzle-Design-Code-master/
                                Vendored Rice/JHU reference source and outputs
Rocket_nozzle_sim_phase2.py     Legacy monolithic prototype; not the primary API
```

---

## Known limitations

1. **No hardware qualification.** All results require independent analysis and
   test evidence.
2. **Constant-property nozzle flow.** CEA does not yet drive variable
   $\gamma$, composition, or transport properties through the MOC field.
3. **Inviscid MOC.** Boundary-layer growth is applied only as a separate
   screen, not coupled into characteristic marching or contour optimization.
4. **Incomplete separation physics.** Empirical onset criteria only — no
   shock/boundary-layer or side-load solver.
5. **Experimental exact-Rao path.** The BVP is mathematically auditable and
   heavily tested, but release-level literature promotion is disabled
   (`BENCHMARK_VALIDATED_AT_RELEASE = False`).
6. **Chart vs. exact-variational angles.** Rao 1960 TOP chart angles are
   parabola-fit design data; the exact variational solution can differ
   systematically. Those deltas are recorded, not forced to zero.
7. **Partial differentiability.** The NASA kernel march and live $\theta_B$
   path are differentiable, but the complete start-line-to-final-wall design
   map and total design derivatives are unfinished.
8. **Low-order thermal/structural models.** Radiation bands, the hydraulic
   graph, CoolProp properties, fatigue, buckling, and CHF remain preliminary
   screens—not qualification CHT, 3-D manifold CFD, cyclic plasticity, creep,
   or FEA.
9. **Preliminary manufacturing CAD.** The full-N channels, ribs, optional
   plenums, and ports are solid-modeled, but bolt holes, injector faces,
   fillets, inserts, welds, tolerances, and process-specific rules are not.
10. **No package metadata or stable API guarantee.** The project runs from its
    source tree and research interfaces may change.

---

## Remaining work

A practical roadmap, grounded in the current code rather than historical phase
labels:

1. Complete and publish the exact-variational chart/reference sweep, define
   accepted exact-vs-TOP-fit deltas, and set the criteria for flipping
   `BENCHMARK_VALIDATED_AT_RELEASE = True`.
2. Make the live differentiable $\theta_B$ solve a routinely exercised
   production path and finish total implicit design derivatives with respect
   to $R_t$, $\varepsilon$, $L_{\%}$, $\gamma$, and ambient pressure.
3. Differentiate or replace the remaining start-line and BDE/final-wall steps
   so sensitivities reach manufacturable wall coordinates, not only the
   control surface.
4. Finish source-port provenance and public reliability wiring for the
   NASA/JHU reference workflow, including the unresolved historical `TT'`
   fixture.
5. Add variable-property frozen/equilibrium thermochemistry and transport
   properties to the nozzle flowfield.
6. Couple viscous displacement, wall pressure, separation, and side-load
   physics to the contour solution; validate against published experiments
   and CFD.
7. Add independent axisymmetric Euler/RANS CFD comparisons with quantitative
   gates for $C_F$, wall pressure, exit Mach, and flow angle.
8. Replace thermal/cooling/structural screens with conjugate heat transfer,
   coolant pressure-drop/property models, temperature-dependent materials,
   and structural analysis.
9. Promote CAD from a revolved screening solid to a manufacturing definition
   with channels, interfaces, fasteners, tolerances, joints, and inspection
   features.
10. Retire or clearly isolate legacy paths, add package metadata, and publish
    a stable API and reproducible release artifacts.

---

## References

Primary literature in [`propulsion_texts/`](propulsion_texts/) and the
reference notes under [`docs/`](docs/) and
[`raosim/benchmark_data/`](raosim/benchmark_data/):

- G. V. R. Rao, *Exhaust Nozzle Contour for Optimum Thrust*, Jet Propulsion,
  1958.
- G. V. R. Rao, *Approximation of Optimum Thrust Nozzle Contour*, ARS J.
  30(6), 1960 — the TOP $\theta_n/\theta_e$ design charts.
- G. V. R. Rao, *Recent Developments in Rocket Nozzle Configurations*, ARS J.
  31(11), 1961 — $\gamma$-insensitivity of the optimal contour
  (`RaoRecentDevinRockNozConfig.pdf`).
- Rao, Beck & Booth, *Rao Variational Optimum Bell Nozzle: A Design
  Compendium*, AIAA 99-2584, 1999 — stationarity Eq. (1)–(2), validity
  boundary (`rao1999.pdf`).
- NASA SP-8120, *Liquid Rocket Engine Nozzles* — chart reproduction,
  separation and wall-pressure guidance.
- Rice, *2D and 3D Method-of-Characteristics Tools for Complex Nozzle
  Development*, JHU/APL RTDC-TPS-481, 2003.
- Kliegel & Levine, *Transonic Flow in Small Throat Radius of Curvature
  Nozzles*, 1969 (`prmeyer.pdf` and related notes).
- J. Östlund, *Flow Processes in Rocket Engine Nozzles…*, KTH, 2002 — Rao
  valid-region discussion (`fulltext01.pdf`).
- R. Stark, *Flow Separation in Rocket Nozzles — An Overview*, 2009.
- J. D. Anderson, *Modern Compressible Flow* — isentropic relations,
  Prandtl–Meyer, MOC compatibility (§11.4).
- M. J. Zucrow & J. D. Hoffman, *Gas Dynamics*, Vol. 2, Ch. 17 — axisymmetric
  characteristics.
- G. P. Sutton & O. Biblarz, *Rocket Propulsion Elements* — performance
  relations.

See [`latex-report/raosim_reference.pdf`](latex-report/raosim_reference.pdf)
for the extended mathematical reference and
[`JAX_DIFFERENTIABLE_PLAN.md`](JAX_DIFFERENTIABLE_PLAN.md) for the
differentiable-solver development record. Those documents contain historical
research notes; **where any status statement conflicts with the executable
code or the current test suite, the code and tests are authoritative.**
