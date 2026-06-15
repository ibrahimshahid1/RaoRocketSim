# RaoRocketSim

RaoRocketSim is an experimental axisymmetric supersonic-nozzle design and
analysis codebase. It contains:

- a NASA/JHU-derived method-of-characteristics kernel and BDE wall march;
- a finite-dimensional Rao stationary control-surface solver;
- NumPy/SciPy and differentiable JAX residual backends;
- reference-data parity, topology, compatibility, and contour tests;
- plotting and export scripts for research-grade nozzle studies.

The generated contours are not hardware-qualified. CFD, thermal,
structural, manufacturing, inspection, and hot-fire validation remain the
responsibility of the user.

## Current formulation

The default solver uses the characteristic formulation with JAX and full
flow-state continuity at point D. With nodes ordered downstream, the
axisymmetric compatibility equations are

```text
C+ (slope theta + mu): d(theta - nu) = -S ds
C- (slope theta - mu): d(theta + nu) = +S ds
S = sin(theta) sin(mu) / r
```

The reference epsilon=10, 80%-length smooth stationary-DE root is near
`theta_B=25.5659 deg` and `kdf=0.15216`.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt pytest
```

JAX is the default `solve_rao_bvp` backend. Pass
`solver_backend="numpy"` to use the SciPy finite-difference regression
path.

## Tests

```bash
.venv/bin/python -m pytest -q -m "not slow"
```

The full suite includes expensive convergence and reference solves:

```bash
.venv/bin/python -m pytest -q
```

## Generate a contour

```bash
PYTHONPATH=. .venv/bin/python scripts/generate_contour.py
```

Outputs are written under `builds/`. See `JAX_DIFFERENTIABLE_PLAN.md` and
`NEXT_SESSION_PROMPT.md` for the detailed derivation and current research
status.
