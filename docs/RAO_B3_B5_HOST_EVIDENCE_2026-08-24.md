# Rao B3/B5 host evidence — 2026-08-24

## Scope and verdict

This note records observed host behavior for the new B3 opt-in post-solve
validator and the B5 $\varepsilon\times L_\%$ declared-domain infrastructure.
It is an implementation evidence record, not a solver qualification report.

- **B3:** the typed host validation contract has landed, but full qualification
  has **not**. The real probes below return `UNKNOWN` because forward-MOC
  compatibility fails before TOP-versus-variational discrepancy adjudication.
- **B5:** the typed declared-domain schema, fail-closed lookup, NumPy
  D-attachment root refinement, kernel-resolution evidence, and Rao Eq. (5)
  classification have landed. The accepted first H2 slice was generated and
  hash-checked, but B5 remains partial beyond that one fixed context and does
  not establish full-BVP convergence.
- **Default path:** `reevaluate()` runs B3 only when given an explicit
  `VariationalPostSolveConfig`. Omitting it adds no variational solve and
  therefore zero added variational-solver runtime to the analytic TOP path.

## Host and timing context

The observations were collected on the project host: Apple M3 Pro (`arm64`),
macOS 15.7.5, with the CPU JAX environment (Python 3.12.13, JAX 0.6.2, NumPy
2.4.6, SciPy 1.17.1). Wall times are individual observations on that host, not
benchmark guarantees and not promises for another machine, cache state, or
solver configuration.

## B3 real-probe record

All three probes reached a numerically converged BVP, an evaluable/valid Rao
Eq. (5) result, and available direct full-C-D-E performance. None is a B3 pass:
the forward-MOC physical-closure gate failed, so the typed outcome is `UNKNOWN`
and the geometry/performance discrepancy is not adjudication-eligible.
Only context fields retained in the run notes are reported; the 12×12 row does
not inherit unrecorded values from the BDE rows.

| Probe | Declared context | Observed wall time | BVP | Eq. (5) | Full C-D-E | Forward-MOC neighbor smoothness | Typed result |
|---|---|---:|---|---|---|---|---|
| 12×12 coupled | $\gamma=1.4$; coupled wall | 16.2 s cold; 12.5 s immediate repeat | pass | pass | available | failed; exact neighbor-Mach maximum was not recorded | `UNKNOWN` |
| 24×24 BDE | $\gamma=1.4$, $P_a/P_0=0.01$, $\varepsilon=10$, $L=80\%$, $R_t=0.020$ m, `physics_weight=1.0`; characteristic formulation, BDE wall, `coupled_wall=False` | 10.93 s | pass | pass | available | neighbor-Mach maximum 0.11810 > 0.10 | `UNKNOWN` |
| 32×32 BDE | same physical/numerical context as the 24×24 BDE row, except resolution | 14.96 s | pass | pass | available | neighbor-Mach maximum 0.13748 > 0.10 | `UNKNOWN` |

The 0.10 neighbor-Mach limit is **repository screening policy**, not a tolerance
stated by Rao 1958/1999 or another qualification authority. Likewise,
`ContourDiscrepancyTolerances` has no implicit defaults: each caller must state
its maximum/rms radius, divergent-length, and thrust-coefficient error policy.
Those discrepancy limits are repository adjudication choices, not literature
qualification limits.

The immediate B3 promotion gate is therefore forward-MOC compatibility on real
solves. Only after that gate passes can the TOP-versus-variational discrepancies
be interpreted against a declared policy. An `UNKNOWN` result must not be
reported as a variational validation pass or replaced silently by TOP geometry.

## B5 fixed-slice artifact, timing, and policy

Observed building-block times on the same host were approximately 1.16 s for an
11×11 internal existence scan, 2.2 s for one D-attachment geometry-root
refinement, and 5.3 s for one 8/12/16 kernel-resolution ladder. These are
component observations, not per-knot guarantees. The accepted fixed-slice policy
adds a 24-point kernel rung. The complete dense 65-knot generation took
approximately 68 minutes on this host; the earlier 11–20 minute planning estimate
was low.

The first H2 slice was generated under this accepted policy:

- external $\varepsilon$ axis:
  `4, 5, 6, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50`;
- external $L_\%$ axis: `60, 70, 80, 90, 100` (65 knots total);
- fixed physical slice: $\gamma=1.4$, $P_a/P_0=0.01$,
  $R_t=0.020$ m, $R_d/R_t=0.382$, and $R_u/R_t=1.5$;
- closure/numerics: NumPy-host smooth D-attachment geometry root,
  Kliegel–Levine start line, 8/12/16/24 kernel ladder, 24 D-E points, mass
  stop, geometry scan, unit residual weights, and maximum Mach 12;
- per-sample evidence: root success and normalized geometry residual, kernel
  convergence in $\theta_B$ and `kdf`, and Rao Eq. (5)
  evaluability/validity/minimum $\mathrm{d}R/R$;
- lookup interpolation policy: maximum $\varepsilon$ spacing 5 and maximum
  length-percentage spacing 10, with all bracketing samples required to be
  converged, to pass the declared kernel-refinement policy, and to be Eq.
  (5)-valid.

The accepted numerical thresholds are a normalized geometry-residual limit of
$10^{-6}$, final-two-rung $\theta_B$ change no greater than 0.05°, and final-
two-rung `kdf` change no greater than $5\times10^{-4}$. These values and the
lookup spacings are **repository policy**, not literature-derived validity
thresholds. They are stored explicitly in the artifact.

### Artifact identity and measured classifications

- path:
  `raosim/data/rao_domains/rao_d_attachment_fixed_slice_gamma1p4_pa0p01_eps_length_v1.json`;
- byte length: 101,932;
- canonical artifact SHA-256:
  `a85de663fffced5559e92760fe933a7edd0ca269fc227aae5908c0eccbf87b19`;
- source model:
  `raosim.rao_existence_scan:epsilon-length-fixed-context-d-attachment-domain:v1`;
- Eq. (5) authority:
  `Rao-Beck-Booth-AIAA-99-2584-1999-pdf-page-4-Eqs-2-3-5-6`.

The JSON is strict and contains no NaN or infinity tokens. Across its 65 knots,
40 are `converged_valid` and 25 are `not_converged`; `converged_invalid`,
`numerical_failure`, `under_resolved`, and `unevaluable` are all zero. Across
the 48 cell midpoints, fail-closed lookup returns 23 `supported`, 11
`refused_not_converged`, and 14 `unknown_under_resolved` results.

### Reproduction command

```bash
.venv-jax/bin/python scripts/rao_existence_scan.py \
  --Rt 0.020 --epsilon 10 --gamma 1.4 --pa-over-p0 0.01 --length-pct 80 \
  --theta-b-center-deg 24 --theta-b-span-deg 10 --theta-b-count 21 \
  --kdf-min 0.005 --kdf-max 0.90 --kdf-count 25 \
  --models smooth --stop-at mass --scan-mode geometry --third-residual auto \
  --n-kernel 8 --n-de-points 24 --root-maxfev 80 \
  --resolution-convergence 8,12,16,24 \
  --domain-theta-seed-policy rao_chart_centered \
  --domain-chart-theta-span-deg 40 --domain-chart-theta-count 21 \
  --domain-epsilon-values 4,5,6,8,10,15,20,25,30,35,40,45,50 \
  --domain-length-pct-values 60,70,80,90,100 --domain-model smooth \
  --domain-residual-limit 1e-6 --domain-theta-resolution-tol-deg 0.05 \
  --domain-kdf-resolution-tol 0.0005 \
  --domain-max-epsilon-spacing 5 --domain-max-length-pct-spacing 10 \
  --domain-output raosim/data/rao_domains/rao_d_attachment_fixed_slice_gamma1p4_pa0p01_eps_length_v1.json \
  --output-dir builds/rao_domain_h2_2026-08-24
```

This first H2 product declares support only for its exact context. It is a
NumPy D-attachment root/kernel-refinement/Eq. (5) domain. Passing its last-rung
change policy is **kernel-resolution evidence only**; it is not proof that every
mesh-dependent quantity is fully resolved, and it is not a universal map of
full variational-BVP convergence. Broader claims require separately generated
and reviewed slices for other $\gamma$, ambient-pressure ratio, throat geometry,
closure model, numerical context, and any fuller mesh-convergence study.

## Software verification

The final B3/B5 tree passed the repository-wide non-slow gate:

```text
.venv-jax/bin/python -m pytest -q -m 'not slow'
2148 passed, 26 deselected, 3 warnings in 1586.04 s
```

The three warnings are the existing Matplotlib animation-lifetime warnings and
CadQuery `save` deprecation warning. The focused Rao/JAX regression slice also
passed 68 tests with 5 deselected. `compileall` and `git diff --check` passed.
A fresh isolated wheel build contains the 101,932-byte domain JSON; reading that
wheel member reproduces the canonical artifact hash and the 40/25
classification counts above.
