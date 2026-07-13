# OpenFOAM VOF Spray Export

`raosim.openfoam` writes deterministic, provenance-bound OpenFOAM cases. It
does not import OpenFOAM, run a solver, or turn a generated dictionary into
validation evidence.

## Pinned solver contract

The first template targets exactly:

- OpenFOAM Foundation 13;
- patch/tag `20260624`;
- `foamRun` with `solver incompressibleVoF`;
- the v13 `phaseProperties` and per-phase `physicalProperties.*` layout;
- MULES interface transport and RAS `realizableKE`.

Every file is UTF-8/LF text with a SHA-256 in
`raosim_openfoam_manifest.json`. The case fingerprint covers the fixture,
geometry, properties, mesh/numerical inputs, boundary flux, and sorted artifact
hashes. It contains no timestamp, host path, random value, or local username.

## The paper has two different CFD stages

Radhakrishnan et al. (2018) must not be read as one simultaneous two-stream
VOF calculation:

1. Section 2.3 injects only water at 300 K into the center gap and uses VOF to
   estimate the liquid-sheet thickness. The surrounding phase is ambient air.
2. Section 2.4 injects the 3.03 g/s annular air stream as the Eulerian carrier
   and injects liquid droplets at the pintle opening for the Lagrangian/WAVE
   calculation.

The implemented profile is named
`radhakrishnan2018_sheet_vof_external_screen`. It therefore does **not** impose
the Table-2 air mass flow. WAVE constants are retained as downstream
provenance, never written into VOF dictionaries.

## Geometry and current fidelity

`MovablePintleVOFGeometry` records every Table-1 dimension in SI:

- post, center-gap, rod, and tip diameters;
- liquid and gas annular gaps;
- post and tip angles and thicknesses;
- post recess;
- mechanical opening;
- the paper's 80 mm axial by 120 mm radial domain.

The first mesh is intentionally a reduced external-gap wedge. It puts the
mechanical opening on a cylindrical radial-water boundary at the tip radius.
It does not yet mesh the internal center-gap turn or the sloped post and pintle
surfaces. Therefore it can exercise external interface transport and sheet
evolution, but it cannot be described as predicting sheet formation or as an
exact reproduction of the paper's internal VOF domain.

The distinction is enforced in the manifest:

- `lopen_mm` is the prescribed mechanical opening;
- author-VOF `sheet_thickness_full_mm` and `water_velocity_m_s` are output
  comparison targets with `prescribed_to_case=false`;
- `internal_injector_geometry_resolved=false`;
- current annulus/slot/hole injectors cannot be relabeled as this movable
  radial-sheet geometry.

## Boundary and flux closure

For wedge angle `theta`, the liquid boundary uses

```text
A_360   = 2*pi*r_tip*L_open
U_r     = mdot_360/(rho*A_360)
mdot_w  = mdot_360*theta/(2*pi)
A_w     = A_360*theta/(2*pi)
```

The curved block edges represent the sector arcs. The manifest independently
records the full-annulus target, wedge target, wedge area, imposed radial
velocity, analytically integrated wedge mass flow, and relative residual.

Water and air density and viscosity, surface tension, inlet turbulence
intensity, and turbulence length scale are explicit inputs. The publication
does not provide the exact Fluent values; repository defaults are labeled as
assumptions, not paper data. `nu=mu/rho` is tested for both phases.

## Export

```sh
lrekit export-openfoam-spray output/openfoam/case_1 --case-row case_1
```

Available benchmark rows are `case_1` through `case_4`. The writer is atomic,
refuses symlinked destination components, accepts an existing case only when
every byte is identical, and never invokes external commands.

The generated `Allrun` checks major version 13, then calls `blockMesh`, strict
`checkMesh`, and `foamRun`. Patch `20260624` must still be verified from the
installed source/package because `foamVersion` reports only the major release.

## VOF-to-parcel handoff

`VOFToLagrangianHandoff` is the typed bridge to
`RadialSheetGeometry`. It carries:

- case, input, solver-build, and extraction fingerprints;
- the averaging window and alpha-isocontour extraction definition;
- full sheet-thickness and velocity means/variation;
- liquid mass and vector-momentum fluxes;
- a fingerprinted axisymmetric carrier field and properties;
- mesh, timestep, domain, and averaging refinement studies;
- additional immutable evidence gates.

Conversion fails until required conservation, coverage, variation, and
convergence gates pass. The converted `sheet_thickness` is always the full
physical thickness, never a half-thickness.

## Evidence still required

The static exporter leaves these gates false:

- exact runtime/tag verification;
- `blockMesh` and strict `checkMesh` evidence;
- solver completion and phase-mass closure;
- mesh, timestep, domain-size, and averaging-window convergence;
- statistically stationary sheet extraction and alpha-threshold sensitivity;
- internal injector geometry fidelity;
- comparison to the 2018 author-VOF sheet thickness and velocity;
- VOF-to-parcel mass and momentum closure;
- experimental SMD, LOX/GCH4, reacting-flow, and hardware validation.

The next geometry increment is a conformal internal movable-pintle wedge using
the full Table-1 passage geometry. A separate two-stream VOF profile may follow,
but it must never be labeled as the paper's Section-2.3 VOF calculation.
