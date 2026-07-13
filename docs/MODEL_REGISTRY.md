# Model Provenance Registry

`raosim.model_registry` is the machine-readable physics/source index for the
integrated design chain. It distinguishes four questions that were previously
mixed together in comments and reports:

1. Is the relation a published equation, a derived balance, or a repository
   heuristic?
2. Where is the local source or source-routing document?
3. What is the physical validity envelope?
4. What level of verification or validation has actually been completed?

Each entry contains:

```text
model_id
subsystem
quantity
relation
source
local_source
equation_ref
validity
status
verification
validation_level
notes
```

`audit_model_registry(repo_root)` fails its software audit when a declared
local source is absent, a published entry has no local source route, required
metadata is empty, or a repository-created assumption is not labeled as a
heuristic/policy/schedule.

The registry covers the principal implemented models in:

- thermochemistry and constant-gamma performance;
- throat discharge and Rao/TOP/BVP/MOC nozzle geometry;
- chamber characteristic-length sizing;
- boundary layer, Bartz, coolant convection, pressure loss, radiation, CHF,
  fin efficiency, liner stress, buckling/fatigue, and material catalogs;
- pintle hydraulics, momentum direction, stability, atomization, evaporation,
  throttle scheduling, and optional spray/c-star coupling;
- pump meanline, inducer/NPSH, synthetic map, electric drive, and battery;
- injector/chamber interfaces, separation/altitude behavior, trajectory, and
  CAD/regen topology checks.

## Meaning of validation levels

Terms such as `software_verified`, `screening`, `reference_matched_subset`, and
`topology_only` are intentionally narrower than `hardware_qualified`.

- **Software verified** means equations, units, identities, trends, and failure
  behavior have automated tests.
- **Benchmark screened/reference matched** means one or more published or
  source-port cases are reproduced inside a declared tolerance and envelope.
- **CFD/CHT/FEA validated** requires independent archived cases and review.
- **Cold-flow/hot-fire validated** requires configuration-controlled test data.
- **Hardware qualified** is never asserted by the model registry.

Physical-release evidence is handled separately by
[`PHYSICAL_RELEASE_GATES.md`](PHYSICAL_RELEASE_GATES.md).
