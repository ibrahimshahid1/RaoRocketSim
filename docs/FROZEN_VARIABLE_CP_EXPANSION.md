# Frozen-composition, variable-$c_p$ nozzle expansion

## Status and claim boundary

[`raosim/frozen_flow.py`](../raosim/frozen_flow.py) implements a
**thermally-perfect, chemically frozen, quasi-one-dimensional** ideal-gas
nozzle expansion. Its model-registry identifier is
`performance.frozen_variable_cp_quasi1d`.

This path removes the calorically-perfect assumption from the one-dimensional
performance calculation: the composition and mixture gas constant remain
fixed, while $c_p$, $c_v$, $\gamma$, enthalpy, sound speed, pressure, density,
velocity, Mach number, mass flux, $c^*$, and $C_F$ are resolved consistently
as functions of static temperature. It does **not** implement equilibrium
expansion, finite-rate chemistry, real-gas flow, CFD, or a variable-property
method of characteristics.

The initial integrated-design applicability is deliberately narrow:

- preliminary Bézier/TOP geometry only;
- fixed-composition ideal-gas property tables with explicit provenance;
- adiabatic, inviscid, isentropic, shock-free, attached quasi-1D flow; and
- software conservation checks, not physical validation.

The implementation must not be described as a CEA equilibrium solver, a
frozen CEA result unless the supplied table actually came from a pinned CEA
artifact, a Rao/MOC solution, CFD validation, or hardware-qualified
performance.

## Governing model

Let the frozen mass fractions be $Y_i$. The mixture molecular weight and gas
constant are fixed throughout the expansion:

$$
R = \frac{R_u}{\overline{M}}, \qquad
c_v(T) = c_p(T)-R, \qquad
\gamma(T) = \frac{c_p(T)}{c_p(T)-R}.
$$

The input table defines a bounded, continuous, piecewise-linear $c_p(T)$. On
one table segment,

$$
c_p(T)=mT+b.
$$

The code integrates this representation analytically rather than applying a
single effective $\gamma$ or numerically summing samples:

$$
h(T_2)-h(T_1)
= \int_{T_1}^{T_2}c_p(T)\,dT
= \frac{m}{2}(T_2^2-T_1^2)+b(T_2-T_1),
$$

$$
\int_{T_1}^{T_2}\frac{c_p(T)}{T}\,dT
=m(T_2-T_1)+b\ln\!\left(\frac{T_2}{T_1}\right).
$$

Integrals that cross table nodes are split into segments. Extrapolation above
or below the declared temperature bounds is rejected.

For chamber stagnation state $(p_0,T_0)$ and zero chamber velocity, steady
adiabatic energy conservation gives the velocity at a static temperature $T$:

$$
h(T_0)=h(T)+\frac{u^2}{2}, \qquad
u(T)=\sqrt{2\int_T^{T_0}c_p(\vartheta)\,d\vartheta}.
$$

The composition is fixed, so $R$ is constant. Isentropy then gives

$$
0=\int_{T_0}^{T}\frac{c_p(\vartheta)}{\vartheta}\,d\vartheta
-R\ln\!\left(\frac{p}{p_0}\right),
$$

or

$$
\frac{p(T)}{p_0}=
\exp\!\left[
\frac{1}{R}\int_{T_0}^{T}\frac{c_p(\vartheta)}{\vartheta}\,d\vartheta
\right].
$$

Static density follows from the ideal-gas equation of state,
$\rho=p/(RT)$. The frozen sound speed and Mach number are

$$
a_f^2=\left(\frac{\partial p}{\partial\rho}\right)_{s,Y_i}
=\gamma(T)RT, \qquad M=\frac{u}{a_f}.
$$

The throat temperature $T^*$ is the bracketed sonic root

$$
u(T^*)=a_f(T^*).
$$

With mass flux $G=\rho u$, the characteristic velocity and area relation are
computed from conservation rather than a constant-$\gamma$ closed form:

$$
c^*=\frac{p_0}{G^*}, \qquad
\frac{A}{A^*}=\frac{G^*}{G(T)}.
$$

For a requested expansion ratio $\epsilon=A_e/A^*$, the solver selects the
supersonic area root. It can also solve the inverse matched-pressure problem:
find $T_e$ such that $p(T_e)=p_e$, then return
$\epsilon=G^*/G_e$.

Finally, ideal thrust is reconstructed directly from exit momentum and
pressure:

$$
C_{F,\mathrm{mom}}=\frac{G^*u_e}{p_0}, \qquad
C_{F,\mathrm{pressure}}=\epsilon\frac{p_e-p_a}{p_0},
$$

$$
C_F=C_{F,\mathrm{mom}}+C_{F,\mathrm{pressure}}, \qquad
F=p_0A^*C_F.
$$

Any separately configured $c^*$ or thrust efficiencies are applied by the
engine layer and must not be folded into these ideal quantities a second time.

## Property and provenance contract

`load_frozen_gas_table()` accepts only a normal, non-symlink UTF-8 JSON file
whose top-level keys exactly match schema version 2. Unknown and missing keys
are rejected. The strict payload records:

- `schema_version` and the model name
  `thermally_perfect_frozen_composition_q1d_v1`;
- `molecular_weight_kg_mol` and normalized
  `composition_mass_fractions`;
- strictly increasing `temperature_nodes_k` and matching positive
  `cp_nodes_j_kg_k`, with $c_p>R$ everywhere;
- `freeze_basis`: `chamber_equilibrium_snapshot`,
  `externally_fixed_composition`, or the test-only
  `manufactured_composition`;
- `composition_state_pressure_pa`, `composition_state_temperature_k`, and
  `mixture_ratio` identifying the state at which composition was frozen (all
  three are mandatory values for a chamber-equilibrium snapshot and may be
  JSON `null` only for the other freeze bases);
- `generator`, `generator_version`, and `thermo_database`;
- a human-readable `source`; and
- `source_artifact_sha256`, the SHA-256 of the upstream property/chemistry
  artifact from which the table was derived.

The loader separately binds the SHA-256 of the JSON input itself. The property
fingerprint includes the mixture, property nodes, freeze state, generator and
source evidence; the expansion fingerprint also includes $p_0$, $T_0$,
$\epsilon$, and $p_a$. This prevents a result produced with one table or
operating point from being silently reused for another.

For `chamber_equilibrium_snapshot`, the expansion chamber pressure and
temperature must match the recorded composition state. This is the intended
rocket-combustion route: equilibrate once at the declared chamber state, then
hold that resulting composition fixed through the nozzle. An
`externally_fixed_composition` table is intended for reference gases whose
composition is independently fixed; it is not a way to erase the
operating-state dependence of rocket-product chemistry.
`manufactured_composition` is reserved for analytic/software test oracles and
must not be promoted as engine-property evidence.

The schema establishes traceability, but a populated string or digest is not
proof that the values are correct. Release evidence still requires inspection
of the upstream artifact and an independently reproducible table-generation
procedure.

## Numerical closures and what they prove

Every solved expansion reports:

- throat sonic residual, $|u^{*2}-a_f^{*2}|/a_f^{*2}$;
- station energy and entropy residuals;
- exit area reconstruction residual; and
- throat-to-exit mass-conservation residual.

The result is rejected when the software closure tolerance is exceeded. Tests
also exercise exact table integrals, sonic and subsonic/supersonic branches,
the matched-pressure inverse, pressure scaling, invalid property ranges, input
fingerprints, and collapse to the existing calorically-perfect solution when
$c_p$ is constant.

These are equation and implementation checks. A near-zero residual is
expected because both sides are computed by the same model; it is not
independent evidence that the property table, ideal-gas assumption, frozen
composition, or one-dimensional flow represents a physical engine.

## Deliberately unsupported couplings

### MOC and Rao/BVP contours

The repository MOC, direct MOC optimizer, NASA source-port workflow, and Rao
variational BVP encode constant-$\gamma$ Prandtl-Meyer functions,
characteristic compatibility equations, critical-Mach definitions,
transonic-kernel relations, and mass/thrust integrals. Replacing the scalar
$\gamma$ with station values would not re-derive those equations and is
therefore forbidden.

The frozen variable-$c_p$ design route is initially Bézier-only. Extending it
to MOC or Rao requires a separately derived thermally-perfect characteristic
formulation and new literature benchmarks, including the variable-property
compatibility development routed through Young's thesis. The existing
constant-$\gamma$ MOC/Rao benchmark status does not transfer to this model.

### Equilibrium and reacting expansion

No species source term is solved and the supplied $Y_i$ are immutable.
Equilibrium recomputation, finite-rate reaction, dissociation/recombination,
multiphase chemistry, and variable molecular weight are outside the model.
Choosing a table made from an equilibrium chamber snapshot makes the *initial
composition* traceable; it does not turn the nozzle solve into equilibrium
expansion.

### Thermal, viscous, discharge, and separation authority

The current gas-side thermal recovery/Bartz path, boundary-layer displacement
model, and Hall throat-discharge correction were developed and tested around
calorically-perfect inputs. They are not made variable-property models by
passing an exit or throat $\gamma$ into them. In particular, a future recovery
temperature must close on enthalpy,

$$
h(T_{aw})=h(T)+r\frac{u^2}{2},
$$

not on the constant-$c_p$ temperature formula.

Until dedicated adapters and benchmarks exist, the variable-property thermal,
Bartz, boundary-layer, and Hall-$C_d$ evidence gates remain failed. The
profile-aware wall-pressure and empirical separation screens may consume the
quasi-1D station profile for diagnostics, but they remain attached-flow
screening relations—not CFD or separated-flow validation.

## Evidence required before promotion

Software readiness and physical/model validation are separate. The following
remain mandatory promotion gates:

1. A configuration-controlled CEA or equivalent thermochemistry fixture with
   generator version, database identity, composition state, complete species
   fractions, upstream artifact hash, and independently checked mixture
   molecular weight and $c_p(T)$.
2. Property-grid refinement evidence showing that throat state, exit state,
   $c^*$, and $C_F$ are insensitive to further temperature-node refinement.
3. Independent published frozen-flow benchmark cases, including both state
   profiles and integrated performance—not only manufactured constant-$c_p$
   collapse.
4. Profile-aware variable-property thermal recovery/Bartz and boundary-layer
   treatment, plus an appropriately re-derived throat discharge model.
5. CFD comparison for shock-free and adverse-pressure-gradient contours,
   followed by controlled cold-flow/hot-fire evidence at the intended
   propellants, pressure, mixture ratio, scale, and ambient condition.

Until these gates pass, validated-design mode must reject this expansion path.

## Local literature map

- Anderson, *Modern Compressible Flow*, 3rd ed.,
  [`5f36b7c4...pdf`](../propulsion_texts/5f36b7c4ded79bb3e90754d0f81682f7a68014be.pdf):
  local-PDF pp. 212-218, Sections 5.2-5.4 (general quasi-1D conservation and
  area-velocity behavior); pp. 669-672, Section 17.3 (high-temperature nozzle
  flow and area ratio); pp. 675-678, Section 17.4 (frozen composition and
  mixture $c_p$); pp. 682-685, Sections 17.5-17.6 (frozen sound speed and the
  limitation of effective-$\gamma$ closed forms).
- Young,
  [`Young_Thesis.pdf`](../propulsion_texts/Young_Thesis.pdf): local-PDF
  pp. 21-23 (frozen versus equilibrium); pp. 46-49 (conservation, equation of
  state, and area); pp. 92-94, including Eq. 4.49 (route toward a future
  variable-property characteristic formulation).
- NASA SP-125,
  [`19710019929.pdf`](../propulsion_texts/19710019929.pdf): local-PDF p. 30
  (frozen-flow definition), pp. 13-19 (performance equations), and pp. 93-95
  (frozen-flow examples).
- NASA SP-8120,
  [`19770009165.pdf`](../propulsion_texts/19770009165.pdf): local-PDF
  pp. 25-29 (temperature-dependent properties, chemistry, and transonic-design
  history).
- Betti et al.,
  [`betti2014.pdf`](../propulsion_texts/betti2014.pdf): local-PDF p. 2
  (CEA chamber composition held fixed in a thermally-perfect CFD context).

Page numbers above identify pages in the locally stored PDF files so reviewers
can reproduce the source audit without relying on an external edition's page
offset.
