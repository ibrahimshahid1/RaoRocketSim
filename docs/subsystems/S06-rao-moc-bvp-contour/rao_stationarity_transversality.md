# Rao Stationarity And Endpoint Transversality

This note fixes terminology for the D-existence scans.

## Fixed-Endpoint Problem In This Repo

The current Rao/MOC BVP is posed with:

- fixed throat radius,
- fixed exit radius,
- fixed nozzle length,
- fixed ambient pressure,
- axisymmetric inviscid constant-gamma flow,
- D attached to a point on the kernel BD curve.

For this fixed-endpoint problem, the hard closure conditions are:

- geometry: `r_E = r_target`, `x_E = L_target`;
- mass: flux through DE equals flux through the selected B-D kernel segment;
- interior Rao stationarity along DE, implemented by the algebraic condition
  `M* cos(theta - alpha) / cos(alpha) = constant`;
- characteristic compatibility along DE.

The endpoint is fixed, so the natural free-endpoint transversality condition is
not automatically a required boundary condition.

## Free-Exit Diagnostic

The NASA/JHU `CalcLRCDE(..., end_condition="rao_free")` branch uses the
free-exit condition

```text
theta_Rao(E) = 0.5 asin(
    2 (p_E / p0 - p_a / p0)
    / ((rho_E / rho0) V_E^2 tan(mu_E))
)
```

The scan reports the corresponding absolute residual

```text
sigma_E = theta_E - theta_Rao(E)
```

in radians. This is useful, but it is not the same statement as fixed
radius/length closure.

## Scan Modes

- `geometry`: stop at mass closure and minimize radius plus length residuals.
- `stationarity`: stop at mass closure and minimize radius, length, and
  absolute `sigma_E`.
- `diagnostic`: preserve geometry closure while reporting `Cf`, exit angle,
  Mach jumps, and other diagnostics.

The quasi-1D thrust coefficient comparison is diagnostic only. A finite-length
bell is not expected to reproduce ideal quasi-1D thrust exactly, so
`performance_residual` is not used as the default hard closure residual.
