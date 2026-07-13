# One-Way Lagrangian Spray Mid-Tier

`raosim.spray` implements the non-reacting Python bridge between a resolved or
declared liquid source and a droplet cloud. It is an opt-in engineering model,
not a replacement for VOF, reacting CFD, or hardware evidence.

## Data flow

```text
explicit source geometry + liquid properties + mdot + injection duration
  <- optional typed VOF sheet handoff (all evidence gates required)
  -> deterministic weighted parcels (mass/momentum closed)
  -> prescribed Eulerian carrier u,rho,mu,T,p,k,epsilon
  -> Schiller-Naumann drag + seeded discrete random walk
  -> versioned WAVE/KH (optional OpenFOAM RT branch)
  -> Eq.16 Spalding evaporation with explicit Sh/D/Bm closure
  -> wall/inlet/outlet/vapor reservoirs + trajectory history
  -> multiplicity-weighted d32, mass percentiles, Rosin-Rammler fit
  -> typed liquid-parcel/gas-carrier cycle handoff
  -> fail-closed coupling gates
```

## OpenFOAM VOF bridge

`raosim.openfoam` now exports a deterministic OpenFOAM Foundation v13
(`20260624`) `incompressibleVoF` wedge and defines the typed
`VOFToLagrangianHandoff`. The first case is deliberately a water-only external
gap screen: the 2018 paper's 3.03 g/s air stream belongs to its later
Lagrangian carrier stage, not its Section-2.3 VOF sheet-thickness solve.

The exporter records the complete movable-pintle Table-1 geometry, but its
current block mesh prescribes the mechanical opening at the external tip radius
and does not resolve the internal turning passage. Author-VOF sheet thickness
and velocity remain output targets. See
[`OPENFOAM_SPRAY_EXPORT.md`](OPENFOAM_SPRAY_EXPORT.md) for the exact mapping,
CLI, hashes, and unresolved gates.

Coordinates are Cartesian SI with `x` on the engine axis. Carrier tables are
axisymmetric `(x,r)` fields; interpolation is bilinear and never silently
extrapolates. Built-in fields and the chamber wall profile receive deterministic
SHA-256 fingerprints so a carrier generated for one mass-flow iterate cannot
be relabeled as another.

## Geometry applicability

Four source forms are explicit:

| Geometry | Initial scale | Primary-path status |
|---|---:|---|
| Movable-pintle radial sheet | full sheet thickness | Radhakrishnan WAVE path available |
| Axial annular sheet | full annular gap | geometric secondary-breakup blob only |
| Rectangular radial slots | hydraulic diameter | geometric secondary-breakup blob only |
| Round radial holes | hole diameter | geometric secondary-breakup blob only |

For the radial sheet, `0 deg` is purely radial and
`u_x=U sin(alpha)`, `u_r=U cos(alpha)`. “Primary-path eligible” only means the
source may enter that breakup formulation. It does not mean cycle eligible.
The injector currently constructed by `raosim.injector` is an axial annulus
plus slots or holes, so its Lagrangian result remains diagnostic until a
geometry-specific primary model or movable-pintle branch exists.

## Published equations and versioning

The radius-based WAVE groups are

```text
We_g = rho_g U_rel^2 a / sigma
Oh   = mu_l / sqrt(rho_l a sigma)
T    = Oh sqrt(We_g)
tau  = 3.726 B1 a / (Lambda Omega)
d_c  = 2 B0 Lambda
```

The wavelength coefficients are separate named choices:

- `reitz_1987`: Taylor coefficient `0.4`, Weber coefficient `0.865`
  (OpenFOAM/Reitz-compatible implementation);
- `radhakrishnan_2018`: `0.4`, `0.87`, exactly as printed in the 2018 paper;
- `radhakrishnan_2021`: `0.45`, `0.87`, exactly as printed in the 2021 paper.

The backward-Euler diameter relaxation follows the OpenFOAM form. Parcel
multiplicity changes as `N_new=N_old(d_old/d_new)^3`, preserving represented
liquid mass and momentum without pretending the publication supplied its raw
daughter-parcel RNG algorithm.

The evaporation microkernel implements the magnitude of 2021 Eq. 16 with a
negative liquid-mass derivative:

```text
dm/dt = -kc A rho ln(1+Bm),   kc = Sh D/d
```

`D`, `Bm`, and the named Sherwood closure (`Sh=2` or Ranz-Marshall) are required
inputs. Droplet temperature, phase equilibrium, real-fluid properties, carrier
energy, and two-way source feedback are not inferred.

## Determinism and conservation

- Source azimuths are deterministic and antipodally balanced.
- Turbulent dispersion uses a local `numpy.random.Generator` with recorded seed
  and bit generator; laminar steps consume no random values.
- Each role receives mass and parcel-momentum reservoir ledgers.
- The equal/opposite carrier impulse is reported as a source demand. Global
  momentum is false until a two-way carrier consumes it.
- Energy is explicitly `not_evaluated_no_droplet_energy_equation`.
- Wall versus outlet assignment uses the first segment boundary crossing.
- SMD sampling planes and the step-interpolation method are serialized; a
  timestep/parcel-count refinement study is required before use.

## Literature fixtures and what “validation” means

Three SHA-pinned fixtures are provided:

- 2018 water/air experimental angle and SMD targets, kept separate from author
  VOF/WAVE values;
- the distinct 2021 water/air Table 5 revision;
- 2021 Tables 3/7/8 LOX/GCH4 inputs and author CFD outputs.

Tables 7 and 8 are not experiments. The publications omit the complete carrier
field, property set, raw parcels, RNG stream, and other inputs required for an
exact end-to-end SMD rerun. `compare_smd_to_benchmark` therefore reports either
an experimental *component-target comparison* or an author-CFD *literature
reproduction* and never promotes either to strict validation. The readiness
report and typed cycle handoff preserve those blockers.

## Minimal standalone use

```python
from raosim.spray import (
    AxisymmetricDomain, LiquidProperties, RadialSheetGeometry,
    SprayMarchConfig, SpraySolverSpec, UniformCarrierField,
    WaveBreakupConfig, initialize_primary_parcels, march_parcels,
)

liquid = LiquidProperties(
    "water", density=997.0, dynamic_viscosity=8.9e-4,
    surface_tension=0.072, temperature=298.0, pressure=2.0e5,
)
source = initialize_primary_parcels(
    RadialSheetGeometry(0.004, 0.0002, 0.0, 40.0),
    role="water", liquid=liquid, mass_flow_rate=0.0229,
    injection_velocity=7.07, injection_duration=0.001, parcel_count=200,
)
result = march_parcels(
    [source],
    carrier=UniformCarrierField(
        velocity=[20.0, 0.0, 0.0], density=1.2,
        dynamic_viscosity=1.8e-5, temperature=300.0, pressure=1.0e5,
    ),
    domain=AxisymmetricDomain.cylinder(
        axial_start=0.0, axial_end=0.1, radius=0.04,
    ),
    solver_spec=SpraySolverSpec(
        time_step=1e-6, maximum_time=0.005,
        parcels_per_liquid_stream=200, eddy_lifetime_constant=0.15, seed=0,
    ),
    march_config=SprayMarchConfig(
        body_acceleration=(0.0, 0.0, 0.0), sampling_planes=(0.05,),
        history_stride=100, mass_tolerance=1e-10, momentum_tolerance=1e-8,
    ),
    breakup_by_role={
        "water": WaveBreakupConfig(
            b0=4.92, b1=0.989,
            coefficient_variant="radhakrishnan_2018",
        )
    },
)
```

The carrier values above are illustrative, not the missing publication field;
this example must not be reported as reproduction of the 2018 experiment.

## Cycle gate status

`build_cycle_handoff` separates gas carrier streams from liquid parcels and
fingerprints the final observation. Lagrangian fixed-point mode requires that
typed handoff; an object containing only `eta_vaporization` is rejected. Today
the following required gates fail by construction:

- phase/critical-state evidence;
- two-way carrier momentum and droplet/carrier energy closure;
- strict target-fluid/geometry benchmark readiness;
- current injector primary-geometry applicability;
- regen/cooling/pump outer-loop closure when regenerative cooling is selected.

Accordingly the existing CLI continues to use only the explicitly labeled
legacy correlation screen. A Lagrangian CLI cycle switch should not be exposed
until those gates can pass.
