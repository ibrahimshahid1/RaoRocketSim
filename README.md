# RaoRocketSim

RaoRocketSim is a Python research and preliminary-design toolbox for
axisymmetric rocket nozzles. It combines a practical Rao thrust-optimized
parabolic (TOP) contour workflow with quasi-one-dimensional performance
estimation, engineering screening models, axisymmetric method-of-
characteristics (MOC) solvers, a finite-dimensional Rao variational boundary-
value problem (BVP), NASA/JHU reference-code ports, and a differentiable JAX
backend.

The default, trusted workflow is the chart-based Rao/TOP quadratic Bezier
contour. The MOC and variational solvers are active research implementations:
they expose residuals, topology, reference comparisons, and reliability
metadata, but they are not yet promoted to design-validated or
hardware-qualified status.

> **Engineering status:** Every generated contour is marked
> `hardware_qualified=False`. Passing the repository's design gates means that
> a result passed internal preliminary screening. It does **not** replace
> independent CFD, conjugate heat-transfer analysis, structural FEA,
> combustion-stability work, material allowables, manufacturing review,
> inspection, proof testing, or hot-fire qualification.

## Current capabilities and maturity

| Area | Current implementation | Status |
|---|---|---|
| Rao/TOP geometry | Upstream and downstream throat arcs plus a quadratic Bezier bell using interpolated Rao chart angles | Trusted preliminary baseline |
| Ideal performance | Constant-\(\gamma\), calorically perfect, quasi-1D isentropic flow; \(C_F\), thrust, \(I_{sp}\), \(c^*\), mass flow, exit state | Preliminary |
| Thermochemistry | Built-in propellant constants or optional RocketCEA chamber \(\gamma\), molecular weight, \(T_c\), and \(c^*\) | Preliminary; downstream flow remains constant-\(\gamma\) |
| Direct MOC wall optimization | Axisymmetric characteristic march coupled to a monotone spline wall and SLSQP/Nelder-Mead optimization | Experimental |
| Rao variational/MOC BVP | NASA-topology seed, Rao stationarity, characteristic compatibility, mass and length closure, D-state continuity, validity and topology diagnostics | Experimental research path |
| Differentiable solver | JAX/Optimistix residual solve, differentiable NASA-style kernel march, optional solved \(\theta_B\), and local \(C_F\) sensitivities | Implemented, research-grade |
| Wall pressure and separation | Quasi-1D wall-pressure estimate and Summerfield, Kalt-Badal, or Schmucker empirical separation checks | Screening only |
| Thermal/cooling/structure | Boundary-layer displacement, Bartz-style heat-flux, rectangular regenerative-channel, and thin-wall hoop-stress screens | Screening only |
| Geometry export | CSV, revolved STL, STEP via CadQuery when available or faceted AP214 fallback; IPT conversion manifest | CAD review input, not production definition |
| Validation | Unit/regression tests, literature manifests, NASA/JHU output parsers, kernel/topology parity checks, and diagnostic reports | Strong software verification; incomplete physical validation |

As of **June 14, 2026**, the normal test selection reports:

```text
784 passed, 4 xfailed, 26 deselected in 350.73 s
```

using:

```bash
MPLCONFIGDIR=/tmp/raosim-mpl \
  .venv-jax/bin/python -m pytest -q -m "not slow"
```

The four expected xfails record known research gaps: unresolved provenance for
one historical NASA `TT'` fixture, an unpackaged Cuffel-Back-Massier dataset,
and literature-promotion tests for the experimental MOC and legacy
variational paths. The 26 deselected tests include long JAX solves, convergence
studies, NASA fixed-end closure, and the full Rao chart sweep.

## What the tool does

For a chamber pressure, ambient pressure, throat size or target thrust,
expansion ratio, propellant model, and contour method, RaoRocketSim can:

- generate a reduced-length Rao/TOP bell, a conical nozzle, an MOC-optimized
  bell, or an experimental Rao variational/MOC contour;
- size throat radius from a requested thrust or size expansion ratio for
  matched ideal expansion;
- calculate ideal exit Mach number and pressure, thrust coefficient, thrust,
  mass flow, effective exhaust velocity, and specific impulse;
- estimate wall pressure, overexpansion separation, altitude performance,
  boundary-layer displacement, heat flux, regenerative-cooling capacity, and
  thin-wall pressure stress;
- generate simplified chamber and convergent geometry from \(L^*\) and
  contraction ratio;
- sweep \(\varepsilon\), \(P_c\), or \(R_t\), compare contour families, and
  run literature-backed benchmark cases;
- plot contours, characteristic nets, Mach/pressure/angle fields, wall
  distributions, exit-plane profiles, topology, residual diagnostics, and JAX
  sensitivity fields;
- export versioned CSV, STL, STEP, JSON, Markdown, and metadata artifacts.

## Physical model

### Assumptions

The core gas-dynamics and nozzle solvers assume:

- steady, inviscid, adiabatic flow;
- a calorically perfect ideal gas with constant \(\gamma\);
- isentropic expansion except where an empirical loss or separation screen is
  applied;
- axisymmetry for the active MOC and Rao implementations;
- a choked throat and a supersonic divergent section;
- no reacting-flow chemistry evolution, finite-rate chemistry, particles,
  film cooling, wall roughness, ablation, shocks, side loads, or fluid-
  structure interaction in the solved flowfield.

The optional CEA integration supplies a chamber-property snapshot. The
`cea_frozen` and `cea_equilibrium` modes currently preserve provenance and
configuration intent, but the nozzle flow is still evaluated with one
effective chamber \(\gamma\); variable-property MOC is not implemented.

The built-in table stores nominal combustion-product properties rather than
raw propellant properties:

| Propellant | \(\gamma\) | \(M_w\) [kg/mol] | \(T_c\) [K] | \(\eta_{Isp}\) | O/F |
|---|---:|---:|---:|---:|---:|
| N2O/Ethanol | 1.22 | 0.0260 | 2800 | 0.92 | 5.5 |
| LOX/RP-1 | 1.23 | 0.0235 | 3400 | 0.96 | 2.6 |
| LOX/LCH4 | 1.24 | 0.0220 | 3500 | 0.96 | 3.5 |
| LOX/LH2 | 1.20 | 0.0100 | 3250 | 0.98 | 6.0 |

Users can also supply custom \(\gamma\), molecular weight, \(T_c\), and
efficiency values.

### Isentropic gas dynamics

For Mach number \(M\) and ratio of specific heats \(\gamma\), the implemented
stagnation-property relations are

$$
\frac{T}{T_0}=\left(1+\frac{\gamma-1}{2}M^2\right)^{-1},
$$

$$
\frac{p}{p_0}=\left(\frac{T}{T_0}\right)^{\gamma/(\gamma-1)},
\qquad
\frac{\rho}{\rho_0}=\left(\frac{T}{T_0}\right)^{1/(\gamma-1)}.
$$

The area-Mach relation is

$$
\frac{A}{A^*}=\frac{1}{M}
\left[
\frac{2}{\gamma+1}
\left(1+\frac{\gamma-1}{2}M^2\right)
\right]^{\frac{\gamma+1}{2(\gamma-1)}}.
$$

RaoRocketSim inverts this relation with Newton iteration on either the
subsonic or supersonic branch. The Prandtl-Meyer angle and Mach angle are

$$
\nu(M)=\sqrt{\frac{\gamma+1}{\gamma-1}}
\tan^{-1}\!\sqrt{\frac{\gamma-1}{\gamma+1}(M^2-1)}
-\tan^{-1}\!\sqrt{M^2-1},
$$

$$
\mu=\sin^{-1}\!\left(\frac{1}{M}\right).
$$

### Performance model

With \(\varepsilon=A_e/A_t\), \(A_t=\pi R_t^2\), and \(A_e=\varepsilon A_t\),
the ideal one-dimensional thrust coefficient is

$$
C_F =
\sqrt{
\frac{2\gamma^2}{\gamma-1}
\left(\frac{2}{\gamma+1}\right)^{\frac{\gamma+1}{\gamma-1}}
\left[1-\left(\frac{p_e}{p_c}\right)^{\frac{\gamma-1}{\gamma}}\right]
}
+\left(\frac{p_e-p_a}{p_c}\right)\varepsilon.
$$

The built-in propellant model applies an empirical efficiency multiplier
\(\eta_{Isp}\) to the complete ideal coefficient:

$$
C_{F,\mathrm{actual}}=\eta_{Isp}C_F,
\qquad
F=C_{F,\mathrm{actual}}p_cA_t.
$$

The characteristic velocity, mass flow, specific impulse, and effective
exhaust velocity are

$$
c^*=\frac{\sqrt{\gamma R T_c}}
{\gamma\sqrt{\left(2/(\gamma+1)\right)^{(\gamma+1)/(\gamma-1)}}},
$$

$$
\dot m=\frac{p_cA_t}{c^*},
\qquad
I_{sp}=\frac{C_{F,\mathrm{actual}}c^*}{g_0},
\qquad
V_e=I_{sp}g_0.
$$

These are ideal-cycle estimates. The efficiency multiplier is not a resolved
loss model and should not be interpreted as a prediction of combustion,
boundary-layer, two-phase, or chemical losses.

## Contour methods

### Conical reference

The conical utility uses the same throat arcs as the bell geometry and a
straight divergent wall at half-angle \(\alpha\). Its length and classical
divergence factor are

$$
L_{cone}=\frac{R_e-R_t}{\tan\alpha},
\qquad
\eta_{div}=\frac{1+\cos\alpha}{2}.
$$

The comparison module also estimates bell divergence loss from an assumed
linear exit-plane flow-angle profile:

$$
\eta_{div}\approx
\frac{\int_0^{R_e}\rho u_x|\mathbf{u}|r\,dr}
{\int_0^{R_e}\rho|\mathbf{u}|^2r\,dr},
\qquad
C_{F,2D}\approx\eta_{div}C_{F,1D}.
$$

This estimate is a comparison aid, not a resolved exit-plane solution for the
Bezier contour.

### 1. Rao/TOP Bezier baseline (`method="bezier"`)

The baseline contour has three pieces:

1. an upstream circular throat arc, normally \(R_u=1.5R_t\);
2. a downstream circular throat arc, normally \(R_d=0.382R_t\);
3. a quadratic Bezier bell from inflection point \(N\) to exit point \(E\).

The exit radius and reference 15-degree cone length are

$$
R_e=R_t\sqrt{\varepsilon},
\qquad
L_{15}=\frac{R_e-R_t}{\tan 15^\circ},
\qquad
L_n=\frac{L_{\%}}{100}L_{15}.
$$

The bell is

$$
\mathbf{B}(t)=(1-t)^2\mathbf{N}+2(1-t)t\mathbf{P}_1+t^2\mathbf{E},
\qquad 0\le t\le1,
$$

where \(\mathbf{P}_1\) is the intersection of the tangent leaving \(N\) at
\(\theta_n\) and the tangent entering \(E\) at \(\theta_e\). The default
angles are bilinearly interpolated from embedded Rao 1960/NASA chart tables
covering approximately \(4\le\varepsilon\le50\) and
\(60\%\le L_{\%}\le100\%\). Inputs outside that grid are linearly
extrapolated by the current interpolator and should be treated cautiously.

This method is the repository's trusted preliminary geometry because it is
deterministic, smooth, endpoint-exact, benchmarked against explicit TOP
geometry, and does not depend on convergence of an experimental flow solver.

### Chamber and convergent geometry

The optional upstream geometry is sized from contraction ratio
\(CR=A_c/A_t\) and characteristic length \(L^*=V_c/A_t\):

$$
R_c=R_t\sqrt{CR},
\qquad
V_c=L^*A_t.
$$

The code subtracts the conical-frustum volume of the convergent section and
assigns the remaining volume to a cylindrical chamber. This is a geometric
volume model only; it does not size an injector, establish residence time,
analyze combustion stability, or design cooling and structural details.

### 2. Direct MOC wall optimization (`method="moc"`)

This path parameterizes the bell as a monotone cubic Hermite spline with
decision variables

$$
\mathbf{q}=[\theta_n,r_1,r_2,\ldots,r_{n_c}].
$$

For each candidate wall, the code builds a transonic starting line, marches an
axisymmetric characteristic net with wall feedback, samples the exit plane,
and minimizes a cost containing negative exit thrust, exit-flow-angle
penalties, radius monotonicity, and curvature regularization. SciPy SLSQP is
used when available; a NumPy Nelder-Mead fallback is retained.

The exported bell is currently reconstructed as a smooth quadratic Bezier
using the optimized entrance and exit angles rather than exporting the sparse
optimization spline directly. This is one reason the method remains
`experimental_moc_geometry`.

### 3. Legacy direct variational path (`method="rao"`)

The legacy Rao path discretizes a supersonic control surface, evaluates thrust,
mass-flow, and length functionals, solves a finite-dimensional constrained
optimization problem, and attempts a control-surface-driven MOC wall
construction. It is retained for regression and comparison, but its public
status is `experimental_variational_geometry`; literature promotion tests are
still expected to fail.

### 4. Rao variational/MOC BVP (`method="rao_variational_moc"`)

This is the main research solver. It uses the NASA/JHU topology

$$
TT'\rightarrow B\rightarrow BD\rightarrow D\rightarrow DE\rightarrow E
$$

to seed a finite-dimensional BVP. Its default configuration uses:

- the JAX/Optimistix Levenberg-Marquardt backend with exact autodiff
  Jacobians;
- the characteristic residual formulation;
- a NASA-style fixed-end topology seed;
- full position, flow-angle, and Mach continuity at point \(D\);
- a continuation ladder that increases weights on mass, length, and endpoint
  constraints;
- separate raw-wall, export-wall, residual, topology, and reliability
  diagnostics.

The solved control-surface unknowns are approximately

$$
\mathbf{u}=\left[M_i,\theta_i,r_i\;\middle|\;
\lambda_2,\lambda_3,\log C,f_D\right],
$$

with optional wall unknowns and an optional live kernel angle \(\theta_B\).
Here \(f_D\) is the arc-length fraction locating \(D\) on the kernel
characteristic \(BD\).

#### Axisymmetric characteristics

With nodes ordered downstream and

$$
S=\frac{\sin\theta\sin\mu}{r},
$$

the implemented compatibility equations are

$$
C^+:\quad \frac{dr}{dx}=\tan(\theta+\mu),
\qquad d(\theta-\nu)=-S\,ds,
$$

$$
C^-:\quad \frac{dr}{dx}=\tan(\theta-\mu),
\qquad d(\theta+\nu)=+S\,ds.
$$

The \(C^+\) control surface \(DE\) is enforced geometrically by reconstructing
its axial coordinates:

$$
x_{i+1}=x_i+
\frac{r_{i+1}-r_i}{\tan(\bar\theta_i+\bar\mu_i)},
\qquad x_1=x_D(f_D).
$$

#### Rao stationarity

The critical Mach number is

$$
M^*=\sqrt{\frac{(\gamma+1)M^2}{2+(\gamma-1)M^2}}.
$$

The primary algebraic optimum-thrust condition is

$$
M^*\frac{\cos(\theta-\alpha)}{\cos\alpha}=C,
\qquad \alpha=\mu=\sin^{-1}(1/M),
$$

implemented in logarithmic form for conditioning. The differential identity

$$
d\ln M^*-(d\theta-d\alpha)\tan(\theta-\alpha)
+d\alpha\tan\alpha=0
$$

is retained as a secondary consistency check.

The older direct-method functionals are also available. Per unit radial
extent, their stagnation-normalized forms are

$$
f_1=2\pi r\left[
\left(\frac{p}{p_0}-\frac{p_a}{p_0}\right)
+\frac{\rho}{\rho_0}\gamma M^2\frac{T}{T_0}
\frac{\sin(\phi-\theta)\cos\theta}{\sin\phi}
\right],
$$

$$
f_2=2\pi r\frac{\rho}{\rho_0}\bar V
\frac{\sin(\phi-\theta)}{\sin\phi},
\qquad
f_3=\cot\phi,
$$

representing axial thrust, mass flow, and axial length.

#### Mass and endpoint closure

Mass closure compares the same surface-normal flux integral on \(DE\) and
the selected \(BD\) segment:

$$
\dot m_\Gamma=\int_\Gamma
2\pi r\,\rho V\left|\sin(\beta-\theta)\right|\,ds,
\qquad
\dot m_{DE}=\dot m_{BD}.
$$

The BVP additionally constrains the commanded exit station and radius,
\(x_E=L_n\) and \(r_E=R_t\sqrt\varepsilon\), and reports scaled residuals for
mass, length, stationarity, characteristic compatibility, geometry,
regularization, penalties, wall endpoints, and wall tangency.

#### Smooth-flow validity

The code evaluates the classical Rao validity inequality

$$
b=1-\frac{d\alpha}{d\theta}
\frac{\tan(\theta-\alpha)+\tan\alpha}
{\tan(\theta-\alpha)-\tan\alpha}\ge0.
$$

If \(b\) becomes negative beyond tolerance, the requested smooth variational
solution is treated as outside the shock-free Rao region and the reliability
level is downgraded.

For the repository's \(\varepsilon=10\), 80%-length, \(\gamma=1.4\) regression
case, the smooth stationary-DE reference is approximately

```text
theta_B = 25.5659 deg
f_D     = 0.15216
D       = (M = 3.40145, theta = 18.5182 deg)
E       = (M = 3.47655, theta = 11.1193 deg)
```

These values are a numerical regression point for the implemented
formulation, not a universal nozzle-design result.

### Transonic starting lines

The MOC code provides:

- `kliegel_levine`: a third-order toroidal-coordinate Kliegel-Levine series;
- `sauer_modified`: a compact leading-order curved-throat approximation;
- `area_ratio`: a quasi-1D area-Mach starting line;
- `hall`: a deprecated alias for `sauer_modified`;
- `nasa_visible_kliegel_levine`: a source-faithful NASA/JHU compatibility
  mode used by the reference port.

The theory-correct Kliegel-Levine implementation documents and tests several
coefficient transcription corrections. A separate source-faithful mode keeps
the historical C++ integer-division behavior required for NASA binary/output
parity. These modes answer different verification questions and are
intentionally not collapsed into one implementation.

## Screening models

The following models are deliberately low-order. They are useful for ranking
concepts and rejecting obviously unsuitable designs, not for qualification.

### Wall pressure

At each contour radius, the code sets

$$
\frac{A(x)}{A_t}=\left(\frac{r(x)}{R_t}\right)^2,
$$

inverts the area-Mach relation, and evaluates \(p(x)/p_c\) isentropically. A
positive downstream pressure increment is flagged as non-monotonic. This is a
quasi-1D estimate, not a boundary-layer or shock solution.

### Separation

Three empirical checks are exposed:

$$
p_{sep}\approx0.4p_a \qquad \text{(Summerfield)},
$$

$$
\frac{p_{sep}}{p_a}\approx\frac{1}{1.88M_e-1}
\qquad \text{(Kalt-Badal implementation)},
$$

$$
\frac{p_{sep}}{p_c}\approx
\left(\frac{p_a}{p_c}\right)^{0.8}M_e^{-1}
\qquad \text{(Schmucker implementation)}.
$$

The first contour station whose quasi-1D wall pressure falls below the chosen
threshold is reported as the estimated separation location. Side loads,
restricted shock separation, free shock separation, hysteresis, and transient
startup are not modeled.

### Boundary layer

The displacement-thickness screen uses a turbulent flat-plate-style
correlation,

$$
\delta^*\approx0.046\frac{s}{Re_s^{1/5}}
\sqrt{\frac{T}{T_w}},
$$

and estimates an effective exit area ratio from \(r_e-\delta_e^*\).

### Heat flux and cooling

The heat-flux model preserves Bartz-like sensitivity to chamber pressure,
characteristic velocity, throat diameter, gas temperature, and wall
temperature, then applies empirical axial and area scaling. It is labeled
`bartz_style_screening`; it is not a full Bartz implementation with resolved
transport properties.

For regenerative cooling, total absorbed heat is estimated by

$$
\dot Q=\int q''(x)\,2\pi r(x)\,dx,
\qquad
\Delta T_c=\frac{\dot Q}{\dot m_c c_{p,c}},
$$

followed by a one-resistance wall/convection temperature estimate. Channel
pressure drop, boiling, coking, critical heat flux, rib conduction, coolant
property variation, and manifold maldistribution are not solved.

### Structural screen

The pressure-stress estimate is the thin-wall relation

$$
\sigma_h\approx\frac{p_c r_{max}}{t_w}.
$$

The reported margins compare this value, estimated wall temperature, and peak
heat flux against user-supplied material limits. Temperature-dependent
allowables, fatigue, creep, weld efficiency, stress concentration, buckling,
and combined loads are outside the current model.

### Atmosphere and trajectory

Altitude performance uses a simplified ISA model through 47 km and an
exponential tail above it. The optional trajectory module integrates a
one-dimensional vertical point mass with

$$
m\dot v=F-\frac12\rho v|v|C_DA-mg(h),
\qquad
g(h)=g_0\left(\frac{R_E}{R_E+h}\right)^2.
$$

It is not currently connected to the main CLI and does not model guidance,
pitch, staging events, changing thrust, winds, or six-degree-of-freedom
dynamics.

## Installation

Python 3.12 is recommended because the differentiable backend is pinned to a
tested JAX stack.

```bash
python3.12 -m venv .venv-jax
source .venv-jax/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pytest
```

Core dependencies are NumPy, SciPy, Matplotlib, JAX, JAXlib, Optimistix, and
Equinox. Optional integrations are not installed by `requirements.txt`:

```bash
# Optional thermochemistry
python -m pip install rocketcea

# Optional true revolved B-rep STEP export; otherwise a faceted STEP is used
python -m pip install cadquery
```

Run commands from the repository root; the project does not currently ship as
an installable Python package.

## CLI usage

### Preliminary Rao/TOP design

CLI pressure units are bar for \(P_c\), kPa for \(P_a\), and millimeters for
\(R_t\) and manufacturing dimensions. Internal APIs use SI units.

```bash
.venv-jax/bin/python main.py \
  --propellant LOX/RP-1 \
  --Pc 45 \
  --Pa 101.325 \
  --Rt 20 \
  --epsilon 10 \
  --length-pct 80 \
  --method bezier \
  --no-plot \
  --output nozzle_profile.csv
```

The output is written to a versioned directory such as
`builds/vNNN_YYYYMMDD_HHMMSS/`, together with `metadata.txt`.

### Size the throat from thrust

```bash
.venv-jax/bin/python main.py \
  --propellant LOX/LCH4 \
  --Pc 60 \
  --target-thrust 10000 \
  --epsilon 12 \
  --no-plot
```

The sizing relation is

$$
R_t=\sqrt{\frac{F_{target}}{\pi C_{F,actual}p_c}}.
$$

### Experimental Rao variational/MOC solve

```bash
.venv-jax/bin/python main.py \
  --propellant LOX/RP-1 \
  --Pc 45 \
  --Rt 20 \
  --epsilon 10 \
  --method rao_variational_moc \
  --rao-moc-n-control 12 \
  --rao-moc-n-kernel 12 \
  --rao-moc-max-nfev 200 \
  --no-plot
```

This path can be computationally expensive and remains experimental. The
`--rao-moc-skip-moc` option skips raw wall/net diagnostics for a faster
residual-only study, but that also prevents MOC reliability promotion.

### Sweep a design variable

```bash
.venv-jax/bin/python main.py \
  --propellant LOX/LCH4 \
  --Pc 60 \
  --Rt 25 \
  --epsilon 10 \
  --sweep epsilon 4 50 20 \
  --no-plot
```

Supported sweep variables are `epsilon`, `Pc`, and `Rt`.

### Run a literature benchmark

```bash
.venv-jax/bin/python main.py \
  --benchmark-case lea_top_schomberg_2014 \
  --benchmark-method bezier \
  --benchmark-report builds/benchmarks \
  --no-plot
```

Available manifests are:

- `lea_top_schomberg_2014`;
- `rao_scarfed_moc_1990`;
- `vulcain_s1_separation_similarity`.

Metrics are explicitly classified as `strict`, `xfail`, or `report` so that
missing physics is recorded instead of being hidden behind a single pass/fail
number.

## Python API

### Baseline contour and performance

```python
from raosim.engine import compute_engine_performance
from raosim.nozzle_geometry import bell_nozzle_contour
from raosim.propellants import get_propellant

prop = get_propellant("LOX/RP-1")
contour = bell_nozzle_contour(
    Rt=0.020,
    epsilon=10.0,
    length_pct=80.0,
    method="bezier",
    gamma=prop.gamma,
)
performance = compute_engine_performance(
    Pc=4.5e6,
    Pa=101325.0,
    Rt=0.020,
    epsilon=10.0,
    prop=prop,
)
```

### Design-gated workflow

```python
from raosim.design import DesignInput, ThermoSpec, design_nozzle_v2

result = design_nozzle_v2(DesignInput(
    thermo=ThermoSpec(
        mode="constant_gamma",
        propellant_name="LOX/RP-1",
    ),
    Pc=8.0e6,
    Rt=0.020,
    epsilon=8.0,
    method="bezier",
    mode="preliminary",
))

print(result.performance.thrust)
print(result.gate_report.to_dict())
print(result.report_sections["thermal"])
```

The API name `ValidatedDesignResult` and workflow mode `validated` mean that
the stricter internal schema and gate set were used. They do not set
`hardware_qualified=True`. Validated mode currently requires RocketCEA, the
Bezier method, regenerative-cooling inputs, material/manufacturing inputs, and
all internal gates to pass.

### Rao BVP and sensitivities

```python
from raosim.rao_variational import RaoSolverConfig, solve_rao_bvp
from raosim.jax import rao_sensitivities

config = RaoSolverConfig(
    Rt=0.020,
    epsilon=10.0,
    gamma=1.4,
    length_pct=80.0,
    solver_backend="jax",
    formulation="characteristic",
)

solution = solve_rao_bvp(config)
sens = rao_sensitivities(config, solution=solution)

print(solution.reliability)
print(solution.residuals.max_scaled)
print(sens.cf, sens.condition_number)
```

The current sensitivity API provides exact derivatives of the control-surface
\(C_F\) with respect to solved node variables, explicit fixed-solution
partials with respect to \(p_a/p_0\) and \(\gamma\), and the residual Jacobian
conditioning. It does **not** yet provide total re-solved derivatives with
respect to \(R_t\), \(\varepsilon\), length percentage, or \(\gamma\), nor a
differentiable sensitivity map on the final bell wall.

## Outputs

Normal CLI runs create `builds/vNNN_YYYYMMDD_HHMMSS/` and may contain:

- contour CSV in meters;
- binary STL inner surface or closed solid;
- STEP solid, using CadQuery when available or a faceted AP214 fallback;
- an Inventor conversion manifest pointing to the authoritative STEP file;
- design-gate JSON and v2 physics-screening sections;
- sweep CSV or benchmark JSON/Markdown reports;
- human-readable metadata with inputs, performance, warnings, gate results,
  and generated filenames.

Native Autodesk Inventor IPT writing is not implemented. Bolt patterns,
injector interfaces, throat inserts, tolerances, weld allowances, and braze
allowances are currently metadata/readiness placeholders rather than modeled
solid features.

## Validation and reference evidence

The repository distinguishes software verification from physical validation.

### Software and mathematical verification

The test suite covers:

- gas-dynamics identities and inverse relations;
- Bezier, conical, chamber, curvature, export, and design-workflow behavior;
- axisymmetric characteristic pairing and source terms;
- transonic Kliegel-Levine coefficients and NASA-visible-source semantics;
- characteristic topology, mass conservation, BDE closure, wall tangency, and
  crossing checks;
- NumPy/JAX primitive and residual parity;
- exact-Jacobian convergence, kernel-march parity, solved-\(\theta_B\)
  derivatives, and \(C_F\) sensitivities;
- literature manifest parsing and strict/report/xfail benchmark policies;
- NASA/JHU legacy-output and Tecplot parsing;
- plotting and diagnostic metadata.

### NASA/JHU reference port

`Three-Dimensional-Nozzle-Design-Code-master/` contains the Rice/JHU 2D MOC,
streamline-tracing, and 3D MOC source package plus sample outputs. The active
Python port implements major pieces of the axisymmetric 2D workflow:

- initial throat line and kernel marching;
- arc wall, interior, special-wall, and axis unit processes;
- mass flow along right-running characteristics;
- point-D selection, point-E integration, fixed-end and free-end closure;
- \(\theta_B\) secant solution;
- BDE-region and wall-contour construction;
- explicit `RaoTopology` objects for \(TT'\), \(B\), \(BF\), \(D\), \(BD\),
  \(DE\), \(E\), and the wall streamline.

The checked-in M3.5 perfect-nozzle wall comparison is within the repository's
\(10^{-3}\) RMS regression gate. However, the historical `TT'.out` generator
provenance is unresolved, and source-port certification is not yet integrated
into the public `solve_rao_bvp` reliability promotion. Consequently, no public
Rao solve should currently be described as `NASA_REFERENCE_MATCHED` solely
because an individual reference test passes.

### Reliability labels

The research code defines the following reliability vocabulary:

- `geometric_approximation`;
- `moc_compatible`;
- `rao_variational_residual_solved`;
- `nasa_reference_matched`;
- `benchmark_validated`;
- `cfd_checked`;
- `experimentally_validated`.

These are an ordered vocabulary, not a claim that every level has been
reached. In the current code, `BENCHMARK_VALIDATED_AT_RELEASE=False`, NASA
promotion is not wired into the public solve, and no contour is CFD-checked or
experimentally validated by this repository.

## Repository layout

```text
main.py                         Primary interactive/batch CLI
raosim/                         Active Python package
  gas_dynamics.py               Perfect-gas and Prandtl-Meyer relations
  nozzle_geometry.py            Rao/TOP Bezier geometry
  engine.py                     Ideal performance model
  design.py                     High-level schemas, gates, and artifacts
  physics.py                    BL, thermal, cooling, structural screens
  moc.py                        General axisymmetric MOC march
  nasa_moc.py                   NASA/JHU-style kernel and BDE topology port
  rao_optimizer.py              Direct MOC wall optimization
  rao_variational.py            Rao functionals and global BVP
  jax/                          Differentiable primitives, residuals, march,
                                solve, and sensitivities
  benchmarks.py                 Literature benchmark runner
  benchmark_data/               Manifests and digitized reference curves
  plotting.py                   Geometry, field, topology, and residual plots
  export.py                     CSV, STL, STEP, and IPT manifest export
tests/                          Unit, regression, parity, and benchmark tests
scripts/                        Research diagnostics and artifact generators
docs/                           NASA provenance, topology, and audit notes
latex-report/                   Mathematical reference report and figures
propulsion_texts/               Local source literature used by the project
Three-Dimensional-Nozzle-Design-Code-master/
                                Vendored Rice/JHU reference source and outputs
Rocket_nozzle_sim_phase2.py     Legacy monolithic prototype; not the primary API
```

## Known limitations

The most important current limitations are:

1. **No hardware qualification.** All results require independent analysis and
   test evidence.
2. **Constant-property nozzle flow.** CEA does not yet drive variable
   \(\gamma\), composition, or transport properties through the MOC field.
3. **Inviscid MOC.** Boundary-layer growth is applied only as a separate
   screen, not coupled into characteristic marching or contour optimization.
4. **Incomplete separation physics.** The tool uses empirical onset criteria,
   not a shock/boundary-layer or side-load solver.
5. **Experimental exact-Rao path.** The BVP is mathematically auditable and
   heavily tested, but release-level literature benchmark promotion remains
   disabled.
6. **Chart versus exact-variational angles.** Rao 1960 TOP chart angles are
   parabola-fit design data; the exact variational solution can differ
   systematically. Those deltas are recorded rather than forced to zero.
7. **Partial differentiability.** The NASA kernel march and live \(\theta_B\)
   path are differentiable, but the complete start-line-to-final-wall design
   map and total design derivatives are unfinished.
8. **Low-order thermal and structural models.** Current gates are screens, not
   CHT, coolant-network, fatigue, creep, or FEA solutions.
9. **Simplified CAD.** Cooling channels, bolt holes, injector faces, inserts,
   welds, tolerances, and detailed manufacturability are not solid-modeled.
10. **No package metadata or stable API guarantee.** The project is run from
    its source tree and research interfaces may change.

## Remaining work

The practical roadmap, based on the current code rather than historical phase
labels, is:

1. Complete and publish the exact-variational chart/reference sweep, define
   accepted exact-versus-TOP-fit deltas, and decide the criteria for setting
   `BENCHMARK_VALIDATED_AT_RELEASE=True`.
2. Make the live differentiable \(\theta_B\) solve a routinely exercised
   production path and finish total implicit design derivatives with respect
   to \(R_t\), \(\varepsilon\), \(L_{\%}\), \(\gamma\), and ambient pressure.
3. Differentiate or replace the remaining start-line and BDE/final-wall steps
   so sensitivities reach manufacturable wall coordinates rather than only the
   control surface.
4. Finish source-port provenance and public reliability wiring for the NASA/JHU
   reference workflow, including the unresolved historical `TT'` fixture.
5. Add variable-property frozen/equilibrium thermochemistry and transport
   properties to the nozzle flowfield.
6. Couple viscous displacement, wall pressure, separation, and side-load
   physics to the contour solution; validate against published experiments and
   CFD.
7. Add independent axisymmetric Euler/RANS CFD comparisons with quantitative
   gates for \(C_F\), wall pressure, exit Mach, and flow angle.
8. Replace thermal/cooling/structural screens with conjugate heat transfer,
   coolant pressure-drop/property models, temperature-dependent materials, and
   structural analysis.
9. Promote CAD from a revolved screening solid to a manufacturing definition
   with channels, interfaces, fasteners, tolerances, joining details, and
   inspection features.
10. Retire or clearly isolate legacy paths, add package metadata, and publish a
    stable API and reproducible release artifacts.

## References represented in the repository

The implementation and verification work draw primarily from:

- G. V. R. Rao, *Exhaust Nozzle Contour for Optimum Thrust* (1958);
- G. V. R. Rao, *Approximation of Optimum Thrust Nozzle Contour* (1960);
- G. V. R. Rao, *Recent Developments in Rocket Nozzle Configurations* (1961);
- Rao, Beck, and Booth, *Rao Variational Optimum Bell Nozzle: A Design
  Compendium*, AIAA 99-2584 (1999);
- NASA SP-8120, *Liquid Rocket Engine Nozzles*;
- Rice, *2D and 3D Method of Characteristic Tools for Complex Nozzle
  Development*, JHU/APL report RTDC-TPS-481 (2003);
- Kliegel and Levine, *Transonic Flow in Small Throat Radius of Curvature
  Nozzles* (1969);
- Anderson, *Modern Compressible Flow*;
- Zucrow and Hoffman, *Gas Dynamics*, Volume 2;
- the literature cases and provenance notes under `propulsion_texts/`,
  `raosim/benchmark_data/`, and `docs/`.

See `latex-report/raosim_reference.pdf` for the repository's extended
mathematical reference and `JAX_DIFFERENTIABLE_PLAN.md` for the detailed
differentiable-solver development record. Those documents contain historical
research notes; when a status statement conflicts with executable code or
tests, the code and current test suite are authoritative.
