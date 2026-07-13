# Physical Release Evidence

RaoRocketSim separates three different claims that must not be conflated:

1. **Software verification** - equations, units, numerical identities, failure
   modes, and serialization have automated tests.
2. **Geometry verification** - a CAD kernel can re-import the artifact as a
   valid solid with the expected scale, volume, interfaces, and clearances.
3. **Physical release evidence** - independent analyses, drawings, inspections,
   proof tests, cold-flow tests, and hot-fire tests have passed under an
   authorized configuration.

Passing the first two levels never sets `hardware_qualified = true`.

## Machine-readable gate

`raosim.release_readiness` defines the mandatory evidence IDs for an engine,
injector, regenerative wall, or pump. A default report is blocked. Evidence is
accepted only through a versioned manifest containing a result, configuration
identifier, artifact reference and SHA-256 digest, reviewer, and ISO review
date for every requirement.

```python
from raosim.release_readiness import (
    evidence_manifest_template,
    load_evidence_manifest,
)

template = evidence_manifest_template("engine")
# Write the template as JSON, complete it through the applicable engineering
# and test processes, then evaluate it:
report = load_evidence_manifest("engine_release_evidence.json")
report.require_complete()  # raises while any evidence is missing/invalid/failed
```

Schema identifier: `lrekit.release_evidence.v1`.

Both trusted entry points include the report automatically. For the CLI, use
`--configuration-id ENGINE-CFG-001-REV-A` with
`--release-evidence-manifest engine_release_evidence.json`; add
`--require-release-evidence` to reject the run before artifact generation when
the evidence set is incomplete or belongs to another configuration. For
`design_nozzle_v2`, set the matching `configuration_id`,
`release_evidence_manifest=Path(...)`, and `require_release_evidence=True`.
Local artifact paths and `file://` references are opened and SHA-256 checked;
archive or HTTPS references remain externally reviewed references because an
offline design tool cannot verify their bytes.

## Engine evidence set

The integrated engine gate requires traceable evidence for:

- chamber/nozzle/injector CFD;
- conjugate heat transfer;
- pressure, thermal, fatigue, buckling, and joint FEA;
- temperature/process/lot-specific material allowables;
- drawings, GD&T, fits, surface finish, BOM, and manufacturing route;
- manufacturing, cleaning, NDE, inspection, and assembly review;
- proof-pressure and leak testing;
- integrated cold-flow distribution and pressure loss;
- ignition/transient/combustion-stability assessment;
- hot-fire pressure, characteristic-velocity, heat-flux, and durability data;
- injector fluid-circuit connectivity and optical cold flow;
- regenerative-channel maldistribution and hydroproof;
- pump fluid-volume connectivity, rotordynamics, and measured performance/NPSH
  maps.

The software reports `evidence_complete` when every record passes. It still
reports `hardware_qualified = false`, because formal qualification belongs to
the responsible engineering authority, not to a numerical design program.

## CAD release rule

STL watertightness or STEP `isValid()` is a topology result only. A machining or
test release additionally needs, as applicable:

- canonical units and configuration/version metadata;
- matching mating features and a passed tolerance stack;
- specified seals, threads, retainers, fasteners, and preload;
- connected extracted fluid volumes with no unintended leak path;
- nonzero classified running clearances and no unclassified interference;
- pressure-rated ports/fittings and accessible assembly/inspection features;
- a process-specific AM, machining/closeout, braze, weld, cleaning, and NDE
  plan;
- proof and leak test acceptance criteria.

Missing requirements are blockers, not warnings that can be converted into a
hardware-ready label by `--allow-unconverged` or another numerical override.
