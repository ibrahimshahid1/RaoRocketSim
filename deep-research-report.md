# Derivation and Optimization of the Rao–Bell Nozzle Contour via Calculus of Variations

## Executive Summary

The modern “bell” rocket nozzle exists because the *ideal* nozzle—one that produces **uniform, parallel exit flow** with **exit pressure matched to ambient**—is typically *too long and heavy* at the large area ratios needed for high-altitude or vacuum performance. citeturn34view0turn39view0 Contouring the diverging wall is therefore a constrained optimization problem: achieve (nearly) ideal thrust while restricting length (and thus mass and vehicle packaging). citeturn34view0turn39view0

In the late 1950s, entity["people","G. V. R. Rao","rocket nozzle designer"] introduced a systematic method that couples **calculus of variations** with the **method of characteristics (MoC)** to compute an **optimum-thrust contoured nozzle** subject to *explicit constraints* (notably nozzle length and a prescribed operating/backpressure condition). citeturn39view0turn15view0turn10view0 The key conceptual move is to optimize not “the wall curve directly,” but the **flow properties along a suitably chosen control surface**; then use MoC to construct a nozzle wall that realizes that flowfield. citeturn10view0turn15view0

In operational engineering practice, the computed optimum contours are frequently replaced by a **near-optimum canted-parabola approximation** (a bell nozzle), because performance is *insensitive* to small deviations from the mathematical optimum and because a single “skewed” parabola can closely approximate many optimum contours for a given length and area ratio. citeturn39view0turn21view0 This is the bridge from “Rao optimum contour” to the widely used “bell nozzle” family.

The optimization is only fully meaningful in **2D/axisymmetric compressible flow**, because in strictly quasi-one-dimensional, inviscid, isentropic theory, nozzle contour does not change the ideal exit state for a given area ratio; the contour matters primarily through **divergence losses, shock formation/separation, and viscous effects**—all intrinsically multidimensional or non-isentropic phenomena. citeturn34view0turn39view0turn21view2

## Historical Context

In the early canonical framework summarized by entity["people","G. V. R. Rao","rocket nozzle designer"] (1961), the thrust of a rocket nozzle is maximized when the nozzle delivers **parallel, uniform exit flow** and **exit pressure equals ambient pressure** (for that operating condition). citeturn34view0 Achieving this at high altitude requires very low \(p_a/p_c\), implying very large area ratios and hence extremely long “uniform exit flow” nozzles if designed in a classical way. citeturn34view0

A simple baseline is the **conical nozzle**, which is short and easy to fabricate, but suffers a thrust decrement because the exit velocity is not purely axial. In a conical-flow approximation, the axial momentum component is reduced by a factor
\[
\eta_{\mathrm{div,cone}} \approx \frac{1+\cos \alpha}{2},
\]
where \(\alpha\) is the cone half-angle. citeturn34view0turn21view2 For a very common compromise \(\alpha \approx 15^\circ\), Rao reports the thrust coefficient decrement compared with one-dimensional theory is about **1.7%**, which is “small enough” that 15° conical nozzles became a length/performance reference. citeturn34view0

The push toward **contoured** nozzles is therefore not mysterious: one seeks a wall shape that turns the exhaust closer to the axis by the exit, reducing divergence loss without paying the full length penalty of an ideal uniform-exit nozzle. citeturn34view0turn39view0 This is the design niche that the “bell nozzle” fills.

By the 1970s design-criteria literature, bell nozzles are described as the standard choice for larger systems because the expansion surface is contoured “for optimum performance within a restrictive length.” citeturn39view0 A common industry shorthand is “**80% bell**,” meaning the bell nozzle length is 80% of the length of a 15° conical nozzle with the same area ratio. citeturn39view0

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["rocket bell nozzle contour diagram","conical nozzle vs bell nozzle comparison diagram","method of characteristics nozzle characteristic net diagram"],"num_per_query":1}

## Modeling Assumptions and Governing Equations

### Flow idealizations and what they buy you

Most “Rao-style” optimum contour derivations begin with the idealizations: **steady**, **axisymmetric**, **inviscid**, **adiabatic**, and typically **isentropic** flow (or at least homentropic in the supersonic core). citeturn10view0turn15view0turn34view0 In this regime, the nozzle is (mathematically) a device that converts stagnation enthalpy to directed kinetic energy with no entropy production—unless shocks appear (which violate isentropy). citeturn39view0

Crucially, the *optimization problem is different* depending on whether you adopt:

- **Quasi-one-dimensional (Q1D) isentropic** analysis: the exit state \((p_e, T_e, M_e)\) is fixed by \(A_e/A_t\) and \(\gamma\), and *contour does not change ideal performance* (it mainly changes losses not present in the model). This is why Q1D is excellent for preliminary sizing but insufficient for contour optimization.
- **Full 2D/axisymmetric inviscid** analysis: contour matters because it controls *flow turning* and thus the distribution of exit flow angles \(\theta(r)\) and hence axial momentum. citeturn34view0turn21view2

The 1976-era bell-nozzle design description explicitly notes that “mathematical optimum” contours come from a **variational-calculus maximization technique**, but that initial conditions (from the transonic solution) and gas properties are approximate; nevertheless, nozzle shape and performance are **not very sensitive** to these approximations. citeturn39view0 This insensitivity is a major reason the parabolic bell approximation is accepted in practice.

### Compressible Euler equations and isentropic closure

For an inviscid compressible gas, the governing equations are the compressible Euler system (written here in conservative form):
\[
\frac{\partial \rho}{\partial t}+\nabla\cdot(\rho\mathbf{u})=0,
\]
\[
\frac{\partial (\rho\mathbf{u})}{\partial t}+\nabla\cdot(\rho\mathbf{u}\otimes \mathbf{u}+p\mathbf{I})=0,
\]
\[
\frac{\partial (\rho E)}{\partial t}+\nabla\cdot\big[(\rho E+p)\mathbf{u}\big]=0,
\]
where \(E=e+\tfrac12|\mathbf{u}|^2\), \(e\) is internal energy per unit mass, and \(p=p(\rho,s)\). Under steady, adiabatic, inviscid, isentropic flow (constant entropy \(s\)), one often closes with a calorically perfect gas \(p=\rho R T\), \(a^2=\gamma p/\rho\). citeturn15view0turn34view0

Useful isentropic relations (Q1D or local thermodynamic closure in MoC):
\[
\frac{p}{p_0}=\left(1+\frac{\gamma-1}{2}M^2\right)^{-\gamma/(\gamma-1)},\qquad
\frac{T}{T_0}=\left(1+\frac{\gamma-1}{2}M^2\right)^{-1},
\]
\[
\frac{A}{A^*}=\frac{1}{M}\left[\frac{2}{\gamma+1}\left(1+\frac{\gamma-1}{2}M^2\right)\right]^{\frac{\gamma+1}{2(\gamma-1)}}.
\]
(These are standard; Rao also uses this structure when relating thrust coefficients to area ratio and \(\gamma\).) citeturn34view0turn35search8

### Thrust and thrust coefficient in 1D vs 2D

A standard performance measure is thrust coefficient
\[
C_F=\frac{F}{p_c A_t},
\]
with chamber pressure \(p_c\) and throat area \(A_t\). Rao writes a one-dimensional thrust coefficient form
\[
C_F = C_{F,v} - \frac{p_a A_e}{p_c A_t},
\]
highlighting the separation between a “vacuum thrust coefficient” and an ambient correction term. citeturn34view0

For a **conical** nozzle, because the exit velocity is not purely axial, Rao gives the conical-flow divergence correction \(\tfrac{1+\cos\alpha}{2}\). citeturn34view0turn21view2 For a **general contoured nozzle**, no single exit-angle formula suffices; one must integrate momentum and pressure across the exit plane accounting for the local flow angle distribution. citeturn34view0 This is exactly the kind of dependence that motivates a variational/MoC approach.

## Variational Formulation and Euler–Lagrange Derivation

### The conceptual structure of Rao’s optimization

The Rao design philosophy (as summarized in later NASA technical reports) is:

1. Choose a nozzle throat/entrance region (often fixed by transonic requirements).
2. Define a **control surface** in the supersonic region (often denoted \(CE\)) across which the flow passes.
3. Express **thrust** and **mass flow** as integrals along that control surface.
4. Maximize thrust subject to fixed constraints (e.g., nozzle length, prescribed mass flow/throat conditions, possibly fixed exit plane location).
5. Use the resulting optimal conditions on the control surface as boundary/target data for the **method of characteristics**, which constructs a shock-free nozzle wall contour producing that flowfield. citeturn15view0turn10view0

NASA’s bell-nozzle monograph states that bell nozzle supersonic contours are produced as a “mathematical optimum” based on a variational-calculus maximization technique. citeturn39view0 The same report emphasizes that small deviations from optimum have small performance consequences, explaining why parabolic approximations are popular. citeturn39view0

### A representative Rao functional (as written in a NASA re-derivation)

A particularly explicit presentation is in a entity["organization","NASA","us space agency"] technical memorandum that modifies Rao’s method; it directly lays out the functional form used for the thrust-maximization problem. citeturn15view0

Consider an axisymmetric control surface \(CE\) parameterized by radius \(r\), with local flow speed \(V\), density \(\rho\), pressure \(p\), flow direction \(\theta\) (measured from the axis), and control-surface inclination \(\phi\). citeturn15view0

Mass flow through a differential element of the control surface is written as
\[
d\dot{m}=\rho V \sin(\phi-\theta)\,dA,
\]
and for an axisymmetric surface \(dA=2\pi r\,ds\) with \(ds=dr/\sin\phi\), yielding an integral of the form
\[
\dot{m}=\int_C^E 2\pi \rho V \,\frac{r\sin(\phi-\theta)}{\sin\phi}\,dr.
\]
citeturn15view0

An integral expression for thrust \(T\) is similarly written over the control surface (combining pressure and momentum-flux contributions). citeturn15view0 A geometric constraint for nozzle length is
\[
L=z_C+\int_C^E \cot\phi\,dr,
\]
and with a fixed throat/entrance region, this is reduced to an integral constraint on \(\int \cot\phi\,dr\). citeturn15view0

Using Lagrange multipliers \(\lambda_2, \lambda_3\) (for mass flow and length constraints), the optimization can be written as maximizing
\[
\mathcal{I}=\int_{r_C}^{r_E}\Big(f_1(M,\theta,\phi,r)+\lambda_2 f_2(M,\theta,\phi,r)+\lambda_3 f_3(\phi)\Big)\,dr,
\]
where, schematically:

- \(f_1\) represents the thrust integrand,
- \(f_2\) represents the mass-flow integrand,
- \(f_3=\cot\phi\) represents the length constraint integrand. citeturn15view0

This is a calculus-of-variations problem with:

- multiple “fields” \(M(r),\theta(r),\phi(r)\),
- a potentially variable endpoint \(r_E\) (depending on whether exit radius is fixed),
- and a Lagrange-multiplier augmentation of equality constraints. citeturn15view0

### First variation and “Euler–Lagrange” stationarity conditions

For a classical functional \(\int L(x,y,y')dx\), stationarity yields the Euler–Lagrange equation
\[
\frac{d}{dx}\left(\frac{\partial L}{\partial y'}\right)-\frac{\partial L}{\partial y}=0.
\]
Here, in Rao’s chosen parameterization, the integrand depends primarily on the variables \((M,\theta,\phi)\) rather than their derivatives. In that special case, the Euler–Lagrange conditions reduce to *algebraic stationarity* of the integrand with respect to each variable, plus endpoint transversality conditions if the upper limit varies. citeturn15view0

Carrying out the first variation (and carefully accounting for which quantities are fixed in the throat-determined region vs free in the optimized region), the NASA re-derivation obtains a structure:
\[
\delta \mathcal{I}=\int_{r_D}^{r_E}\Big[
\big(f_{1,M}+\lambda_2 f_{2,M}+\lambda_3 f_{3,M}\big)\delta M
+
\big(f_{1,\theta}+\lambda_2 f_{2,\theta}+\lambda_3 f_{3,\theta}\big)\delta\theta
+
\big(f_{1,\phi}+\lambda_2 f_{2,\phi}+\lambda_3 f_{3,\phi}\big)\delta\phi
\Big]\,dr
+\Big(f_1+\lambda_2 f_2+\lambda_3 f_3\Big)\Big|_{E}\delta r_E,
\]
and since \(\delta M,\delta\theta,\delta\phi,\delta r_E\) are arbitrary in the optimized segment, their coefficients must vanish. citeturn15view0

Thus the interior stationarity conditions are:
\[
f_{1,M}+\lambda_2 f_{2,M}+\lambda_3 f_{3,M}=0,
\]
\[
f_{1,\theta}+\lambda_2 f_{2,\theta}+\lambda_3 f_{3,\theta}=0,
\]
\[
f_{1,\phi}+\lambda_2 f_{2,\phi}+\lambda_3 f_{3,\phi}=0,
\]
and the endpoint (transversality) condition for a free \(r_E\) is
\[
\Big(f_1+\lambda_2 f_2+\lambda_3 f_3\Big)\Big|_{E}=0.
\]
citeturn15view0

Because \(f_3=\cot\phi\) does not depend on \(M\) or \(\theta\), one can eliminate the multipliers from the first two equations and obtain a reduced optimality condition of the form
\[
\frac{f_{1,\theta}}{f_{2,\theta}}=\frac{f_{1,M}}{f_{2,M}}.
\]
citeturn15view0

This is the “variational core” that determines what control surface and what distribution of flow variables along it are compatible with maximum thrust under the stated constraints.

### Connection to isentropic relations and MoC compatibility

To turn the abstract stationarity conditions into computable relations, one expresses \(p,\rho,V\) as functions of \(M\) using isentropic relations (or equilibrium chemistry tables in more advanced variants). The NASA re-derivation explicitly differentiates these relations with respect to Mach number to obtain expressions like \(dp/dM\), \(d\rho/dM\), and \(dV/dM\), which are then substituted into the \(f_{i,M}\) and \(f_{i,\theta}\) terms. citeturn15view0

At that point, the optimality condition becomes a relationship among \((M,\theta,\phi)\) along the optimal control surface. The nozzle wall is then built using the **method of characteristics**—a standard tool for 2D/axisymmetric isentropic supersonic flow. citeturn15view0turn24search13

A compact MoC statement (ideal-gas, 2D or axisymmetric) is that information propagates along characteristic families, and the flow-turning and Mach variations satisfy compatibility relations. The entity["organization","MIT OpenCourseWare","course materials platform"] notes explicitly give the characteristic-compatibility structure and show the **axisymmetric modifications** compared with the planar case. citeturn24search13

### Mermaid flowchart of the derivation workflow

```mermaid
flowchart TD
  A[Choose modeling regime\nsteady, axisymmetric, inviscid,\n(iso/homentropic) supersonic core] --> B[Define geometry + throat/entrance solution\n(transonic region fixes starting data)]
  B --> C[Pick control surface CE in supersonic region\nunknown inclination φ(r)]
  C --> D[Write integral expressions\nmass flow ṁ(CE), thrust T(CE), length L(CE)]
  D --> E[Form augmented functional\nI = ∫ (f1 + λ2 f2 + λ3 f3) dr]
  E --> F[Compute first variation δI\ninclude free endpoint → transversality]
  F --> G[Stationarity conditions\n(algebraic EL + endpoint condition)]
  G --> H[Use isentropic relations\n(p,ρ,V as functions of M)\nreduce conditions to relations among (M,θ,φ)]
  H --> I[Construct flowfield + wall\nvia Method of Characteristics\n(satisfy slip wall + target exit conditions)]
  I --> J[Parabolic/bell approximation\nfit near-optimum contour for engineering use]
  J --> K[Validate + iterate\nCFD / boundary layer / separation constraints\noptimize with constraints if needed]
```

## Boundary Conditions, Transversality, and Shock/Separation Constraints

### Throat and entrance region constraints

The nozzle throat region is not “free” in most practical optimizations: it is constrained by transonic flow behavior, cooling, and geometry. Rao’s 1961 review separates the nozzle into convergent, throat, and divergent supersonic regions and notes that different analysis methods apply in each; mass flow is largely fixed by throat area and chamber conditions. citeturn34view0 The 1976 design-criteria monograph similarly emphasizes that initial conditions for the supersonic contour come from a transonic solution, and while these are approximate, the resulting optimum contour is not very sensitive to them. citeturn39view0

Mathematically, this “fixed entrance” idea appears in the variational setup as a segment of the flowfield (near the throat) where variations \(\delta M,\delta\theta,\delta\phi\) are taken to be zero because the throat solution is predetermined. citeturn15view0

### Exit conditions and endpoint transversality

Two common design ideals are:

- **Parallel exit flow** (flow angle \(\theta \to 0\) at the exit plane), minimizing divergence loss.
- Exit pressure near ambient for design altitude (in the 1D ideal), to avoid pressure-thrust mismatch. citeturn34view0

In variational language, whether the exit radius/location is *fixed* or *free* changes the boundary conditions. If the upper integration limit (exit radius on the control surface) is variable, you obtain an endpoint condition of the form \((f_1+\lambda_2 f_2+\lambda_3 f_3)|_E=0\), i.e., a transversality constraint coupling the state at the exit to the multipliers. citeturn15view0

### Shock-free assumption and where it breaks

Rao-style optimum contours are typically constructed as **shock-free isentropic expansions** in the design condition by MoC. citeturn15view0turn24search13 But real engines must also survive *off-design*:

- At sea level, large-area-ratio altitude nozzles are often **overexpanded**; separation may occur and can produce severe side loads. citeturn39view0turn21view4  
- The 1976 monograph gives a cautionary example: a nonoptimum parabolic contour selected to raise exit wall pressure experienced a wall-pressure minimum, causing unstable asymmetric separation during startup and structural failures; mitigations (diffuser, restraining arms) were required. citeturn39view0  
- It also notes that separation prediction is uncertain, multiple “rules of thumb” exist, and gives an empirical fit-based separation criterion for short contoured nozzles. citeturn39view0  

This is the practical reason many optimizations include additional inequality constraints such as:

- monotonic wall pressure decrease to avoid adverse gradients, citeturn39view0  
- minimum wall pressure bounded away from separation pressure, citeturn39view0  
- robustness to nozzle pressure ratio excursions (startup/shutdown), e.g., avoiding regimes that trigger asymmetric separation and side loads. citeturn21view4  

## Solution Approaches and Numerical Methods

### Analytical structure: what can be solved “on paper”

Two analytical layers are typically distinguishable:

1. **Variational optimality layer:** yields algebraic stationarity relations among state variables on a boundary/control surface (plus a transversality condition if needed). citeturn15view0  
2. **Hyperbolic PDE layer (supersonic flowfield):** solved by MoC, with characteristic compatibility relations. citeturn24search13turn34view0  

The second layer is rarely closed-form globally, because the nozzle is a boundary-value problem with geometric constraints; however, individual elements (Prandtl–Meyer expansions, characteristic marching rules, centerline/wall-point unit processes) have analytic formulas embedded inside numerical marching. citeturn24search13

### Classical numerical methods used in Rao-style design

The classical computational pipeline is a “hybrid analytic–numeric” scheme:

- Solve/assume a throat/starting line (transonic region).
- Use stationarity to determine conditions along a control surface.
- March the characteristic net between the starting line and the control surface, and generate the nozzle wall as a slip boundary (streamline) consistent with the characteristic net. citeturn15view0turn24search13  

Within this workflow, common numerical techniques include:

**Shooting / marching (MoC):**  
Characteristic marching is inherently a shooting-like method: you iterate on boundary parameters so that downstream conditions (e.g., exit angle uniformity, exit Mach distribution, matching to the variationally determined control surface) are met. The MIT MoC notes illustrate this as a sequence of unit processes (internal point, wall point, etc.), with added coupling for axisymmetric geometry. citeturn24search13

**Finite-difference / collocation for the variational stage:**  
If one chooses to parameterize \(M(r),\theta(r),\phi(r)\) along the control surface explicitly, the stationarity conditions can be enforced at discrete points (finite-difference in \(r\)) or via collocation, together with integral constraints (mass flow, length). This becomes a nonlinear system.

**Constrained nonlinear optimization (direct methods):**  
Later practice often parameterizes the wall contour directly (e.g., splines, arcs, Bézier curves) and computes performance by CFD or reduced-order models, then uses constrained optimization. The NASA bell-nozzle monograph explicitly notes “cut-and-try” optimization using available 3D flow programs within program geometric limits, plus extensive cold-flow testing for 3D nozzles. citeturn39view0

### “Perfect bell nozzle” optimization sets as a practical design space

A valuable complementary dataset is the entity["organization","NASA","us space agency"] report on “perfect bell nozzle” parametric optimization curves. It defines “perfect” (wind-tunnel) bell nozzles as axisymmetric nozzles constrained so that at an untruncated design area ratio the exit velocity vectors are uniform and parallel. citeturn22view0 It distinguishes:

- gross thrust coefficient \(C_{F,g}\) (includes wall pressure contribution),
- net thrust coefficient \(C_{F,n}\) (subtracts estimated wall shear/drag),
- ideal one-dimensional coefficient \(C_{F,i}\),
and defines a nozzle efficiency factor \(\eta_n=C_{F,n}/C_{F,i}\). citeturn22view0

This report also emphasizes that “minimum length perfect bell nozzles” are close in contour and performance to minimum-length nozzles designed by the Rao method, but differ in constraint details. citeturn22view0 This is practically useful: it gives engineers a constrained design space and trade curves for length/area ratio/efficiency without reproducing every variational detail.

### Sensitivity to backpressure and Mach number

Backpressure sensitivity is fundamentally about whether the nozzle internal pressure distribution can adjust without triggering shocks/separation. The 1976 bell-nozzle monograph stresses that overexpansion during ground testing can drive flow separation and large asymmetric loads; separation prediction methods are uncertain and used mainly as guides. citeturn39view0

A major modern refinement is recognizing **separation mode transitions** (free shock separation vs restricted shock separation) as key contributors to side loads. A detailed thesis focused on separation and side loads describes side-load models based on momentum balance over the nozzle surface and compares model predictions to experiments, reporting agreement on the order of a few percent for certain cases. citeturn21view4

## Comparison of Contours and Performance Implications

### Geometry and primary performance mechanisms

The table below summarizes how contour choice affects length and performance in the modeling hierarchy. “Performance” is separated into the ideal 1D component (set mostly by \(A_e/A_t\) and \(\gamma\)) and the multidimensional “real nozzle” losses (divergence, shocks, viscous drag).

| Contour type | Primary geometric definition | Primary loss mechanisms captured in 2D/real flow | Typical role in design |
|---|---|---|---|
| Ideal uniform-exit nozzle | MoC contour producing uniform, parallel exit flow and \(p_e=p_a\) at design | Minimal divergence; long length/weight dominates practicality | Theoretical upper bound; reference for maximum thrust citeturn34view0 |
| 15° conical nozzle | Straight-wall divergent cone, half-angle \(\alpha\approx15^\circ\) | Divergence loss; possible shocks from geometry transitions; viscous losses | Baseline/reference; cheap and robust; moderate performance penalty citeturn34view0turn39view0 |
| Rao optimum contoured nozzle | Derived by variational thrust maximization + MoC construction | Designed to reduce divergence for limited length; still subject to off-design shocks/separation | High-performance contour under explicit length constraints citeturn39view0turn15view0 |
| Bell / “canted parabola” approximation | Skewed parabola with prescribed initial and exit wall angles; length often specified as % of 15° cone | Slightly suboptimal vs exact optimum; still sensitive to separation if overexpanded | Engineering standard: near-optimum performance with reduced length citeturn39view0turn21view0 |

### Quantitative checkpoints from authoritative sources

**Conical divergence loss and the 15° reference.**  
For a conical nozzle, Rao gives the conical-flow divergence factor \(\eta_{\mathrm{div,cone}}=(1+\cos\alpha)/2\). citeturn34view0 At \(\alpha=15^\circ\), \(\eta_{\mathrm{div,cone}}\approx0.983\), consistent with Rao’s statement that the one-dimensional thrust coefficient decrement is about **1.7%**. citeturn34view0

**Bell nozzle length definition.**  
“80% bell” means bell length is 80% of the length of a 15° cone with the same expansion ratio. citeturn39view0 This convention is extremely helpful when comparing weight/packaging, because—in first approximation—nozzle mass scales strongly with length and wetted area.

**Why “any 80% parabola” is not automatically better.**  
A detailed study focused on nozzle flow processes cautions that it is a common misunderstanding that any 80% parabolic bell nozzle always yields increased performance over a 15° cone. It cites a case at expansion ratio 100 where an arbitrarily chosen 80% parabolic nozzle yielded only about **0.07% higher inviscid specific impulse** than the conical nozzle. citeturn21view0turn21view1 This is a practical warning: *the details of the parabolic angles/shape matter*, and “bell” is not a magic word—it is a carefully tuned approximation to an optimized contour.

### Practical construction of the bell (parabolic) approximation

The 1976 bell-nozzle monograph states that for a given area ratio and length, a single **canted parabola** can closely approximate the family of optimum contours across different chamber conditions, and that near-optimum parabolic contours can be generated without a computer. citeturn39view0 In practice, the design workflow is:

1. Choose design constraints: \((A_e/A_t)\), percent length (e.g., 80%), allowable exit angle, separation constraints.
2. Obtain recommended initial and final wall angles from optimization curves (the monograph explicitly points to such curves). citeturn39view0
3. Fit a parabola (often “skewed”/canted relative to the axis) satisfying:
   - point continuity at the throat/inflection region,
   - prescribed slopes at start of parabola and at exit,
   - prescribed length and exit radius.
4. Validate with MoC or CFD and adjust to maintain monotone wall pressure where separation risk is critical. citeturn39view0turn24search13

### Performance versus exit Mach number

The plotted curves above show the **ideal vacuum thrust coefficient** \(C_{F,vac}\) as a function of exit Mach number \(M_e\) for representative \(\gamma\) values. This is logically upstream of contour choice: contour choice mainly affects how closely the real nozzle achieves the ideal exit state with minimal divergence and losses.

Two important takeaways:

- Increasing \(M_e\) increases \(A_e/A_t\) rapidly (log-scale plot), meaning performance gains push you toward longer/larger nozzles unless you accept truncation/contouring tradeoffs.
- \(C_{F,vac}\) grows with \(M_e\) but with diminishing returns at large \(M_e\), so “perfectly expanding to extremely high Mach” yields progressively smaller incremental benefit per added expansion ratio/length.

These trends are consistent with the broader engineering narrative in Rao’s review and NASA design criteria: ideal nozzles become excessively long at high area ratio, motivating contoured and truncated designs. citeturn34view0turn39view0

## Recommended References

### Foundational original sources

- entity["people","G. V. R. Rao","rocket nozzle designer"], “Recent Developments in Rocket Nozzle Configurations,” *ARS Journal* (1961). (Accessible scan includes conical divergence factor, length comparisons, and design context.) citeturn34view0  
- entity["people","G. V. R. Rao","rocket nozzle designer"], “Exhaust Nozzle Contour for Optimum Thrust,” *Journal of Jet Propulsion* (1958). (Original variational/MoC optimum-thrust contour paper; cited by multiple NASA sources as the basis of the method.) citeturn15view0turn39view0  
- entity["people","G. V. R. Rao","rocket nozzle designer"], “Approximation of Optimum Thrust Nozzle Contour,” *ARS Journal* (1960). (Key source behind the parabolic/bell approximation curves as referenced in later design criteria.) citeturn39view0turn21view0  

### Authoritative NASA and thesis references used heavily in this report

- entity["organization","NASA","us space agency"] Space Vehicle Design Criteria monograph, “Liquid Rocket Engine Nozzles” (1976/1977 scan). (Defines 80% bell, discusses variational-calculus optimum contour programs, parabolic approximation, and separation risks.) citeturn39view0  
- entity["organization","NASA Lewis Research Center","Cleveland, OH, US"] technical memorandum describing a Rao-method formulation and its variational conditions in detail (1990). citeturn15view0  
- entity["organization","NASA","us space agency"] report on “Perfect Bell Nozzle Parametric Optimization Curves” (1983). (Defines thrust coefficients and efficiency measures; provides optimization trade sets; notes similarity to Rao minimum-length nozzles.) citeturn22view0  
- entity["organization","KTH Royal Institute of Technology","Stockholm, Sweden"] dissertation on flow separation and side loads in rocket nozzles (2002). (High-value for separation physics, side-load mechanisms, and nuanced comparisons for parabolic vs conical performance.) citeturn21view0turn21view2turn21view4  

### Method of characteristics references suitable for rigorous derivations

- entity["organization","MIT OpenCourseWare","course materials platform"] lecture notes on nozzle design by the method of characteristics (includes axisymmetric modifications and characteristic relations). citeturn24search13  

### Practical guidance on what still needs “specific values”

For any fully numerical, design-grade comparison (thrust coefficient, \(I_{sp}\), mass) you must specify at least:

- chamber stagnation conditions \(p_c, T_c\) (or equivalently \(c^*\)),
- gas model (\(\gamma\) constant vs equilibrium/frozen/finite-rate chemistry),
- target operating ambient (design altitude/backpressure trajectory),
- manufacturing/cooling constraints that set wall thickness and thus weight scaling,
- acceptable separation/side-load envelope for startup/shutdown. citeturn39view0turn21view4