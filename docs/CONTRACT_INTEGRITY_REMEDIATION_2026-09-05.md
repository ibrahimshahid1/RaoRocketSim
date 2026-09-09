---
title: Contract integrity remediation
subtitle: W01–W04 — one defect with four faces, and how to close it
version: "0.1"
audience: [ibrahim, AI coding agents]
created: 2026-09-05
companions:
  - docs/audit_work/LREKIT_WORKFLOW_AUDIT_2026-09-04.md   # where W01-W06 were found
  - docs/LREKIT_HARDWARE_PROGRAM_2026-09-05.md            # this is stage 0-1 of that program
  - docs/INTEGRATION_PROGRAM_2026-08-22.md                # §2.1, the configuration object
---

# Contract integrity remediation

> Every file:line below was re-verified against the working tree on 2026-09-05.
> Implementation update, September 9: request/solver/unit/report repairs are now in the working tree. See [Input and execution integrity](INPUT_EXECUTION_INTEGRITY_2026-09-09.md) for implemented behavior, regression evidence and remaining work. The detailed reproductions below describe the September 5 baseline.

---

## 0. These are not four bugs

They present as four unrelated defects. They are one, and seeing that is what makes
the fix tractable — because four independent patches would each add a special case,
and the pattern would grow a fifth face next month.

**The defect: parsing is not binding.** In every case the repository builds a
correct, typed, validated object describing what the user asked for — and then the
next stage does not take it as input. The typed object is *advisory*. Three
mechanisms, in order of increasing depth:

| | Mechanism | Where the value dies | Findings |
|---|---|---|---|
| **A** | **Dropped at the resolver boundary.** The field is parsed into the request object; the resolver never reads it. | `resolve_engine_job` | W01 (altitude, isp_min), W03 (architecture, output policy, schema version) |
| **B** | **Computed, then discarded at the solver boundary.** The resolver does the full work; the solver's entry point has no parameter to receive it and re-derives its own. | `solve_min_mass` | W02 |
| **C** | **Asserted without evidence at the artifact boundary.** The stage receives a path and infers a claim that only the producer could have known. | `build_engine_parts` | W04 |

The fix has one shape in all three: **make the resolved object the only thing the
next stage can take, and delete the parameter that lets it re-derive.** Not
"discouraged by convention" — *structurally impossible*, because the alternative
input no longer exists in the signature.

This is why these four sit at stage 0–1 of the hardware program. They are the
configuration-object work (`INTEGRATION_PROGRAM` §2.1, Track K0) seen from the input
end. Solving them separately from K0 would mean building the same discipline twice.

---

## 1. W01 — accepted mission conditions do not all reach the solve

### What actually happens

`MissionRequest.resolved()` (`raosim/mdo/job.py:293`) is thorough. It validates
`thrust_condition ∈ {sea_level, vacuum, altitude}`, **requires** an altitude when the
condition is `altitude`, parses it dimensionally, and parses five optional
requirements — `qualification_duration`, `isp_min`, `dry_mass_max`,
`envelope_diameter_max`, `envelope_length_max` — even cross-checking that
qualification duration is not below flight duration.

Then `resolve_engine_job` (`raosim/mdo/job.py:660`) builds the mission like this:

```python
mission = MissionSpec.for_propellant(
    spec.architecture.propellant, requirement["thrust"]
)
overrides: dict[str, Any] = {"burn_time": requirement["burn_duration"]}
if requirement["thrust_condition"] == "vacuum":
    overrides["Pa"] = 0.0
```

**Four keys reach the mission**: propellant, thrust, burn time, and `Pa = 0` for the
vacuum special case. `requirement["altitude"]` — validated, dimensioned, required by
the parser — is never read. So `thrust_condition: altitude, altitude: 10 km`
resolves to `mission.Pa == 101325.0`, identical to the sea-level default, with no
note saying a condition was ignored.

This is not a reporting-metadata gap. Ambient pressure sets the delivered thrust
coefficient, and the throat area is sized from thrust — so the altitude case silently
designs a different engine than the one requested.

`isp_min` has the same shape by a different route: `_isp_floor`
(`raosim/mdo/execute.py:241`) walks `spec.targets` only, so a mission-level
`isp_min` never becomes an optimizer constraint.

### The fix

Two parts, and the second matters more than the first.

**(a) Map the field.** `raosim.atmosphere.pressure(h)` already exists
(`raosim/atmosphere.py:72`). The altitude case is one line in the override dict plus
a resolution note recording which atmosphere model produced the pressure — the model
identity belongs in the record, not just the number.

**(b) Make dropping impossible.** The reason altitude was lost is that the override
dict is *hand-assembled*, so a field is included by someone remembering to include
it. Replace it with an explicit total mapping: a table from every
`MissionRequest.resolved()` key to exactly one of

- a named `MissionSpec` field,
- a target/constraint compilation,
- an evidence requirement (e.g. `qualification_duration` is a test requirement, not a
  solve input), or
- an explicit `unsupported` refusal citing the existing unavailable-metric reason.

Then assert at resolution time that the mapping's domain equals the resolved key set.
A new field added to `MissionRequest` without a mapping entry **fails the resolver**,
rather than being silently ignored — which is the property that keeps this fixed.

`dry_mass_max` and `envelope_*` are the interesting cases: `raosim/mdo/metrics.py:178`
already catalogues whole-engine dry mass and installed envelope as unsupported. They
should compile to an explicit refusal quoting that reason, **not** be parsed and
dropped.

### The test that proves it

Parameterised over every field of `MissionRequest`: setting the field changes the
resolved output, or produces a stated `unsupported` result. Plus the specific
equivalence: `thrust_condition='altitude', altitude='0 m'` and
`thrust_condition='sea_level'` resolve to the same `Pa`; at `10 km` they differ by the
atmosphere model's value.

**Cost: small.** No structural blocker.

---

## 2. W02 — the optimizer discards declared pins and bounds

This is the one with a real structural blocker, and it is worth being precise about
where it sits, because the obvious fix does not work.

### What actually happens

The resolver does the whole job properly. `raosim/mdo/job.py:726-770` builds
`space` from `default_design_space(mission)`, walks each variable against the user's
stated policy, and produces four collections:

- `fixed` — pinned values
- `bounds` — per-variable search bounds
- `enumerations` — discrete choices
- `active` / `defaulted` — which variables the optimizer owns, and which took a
  reference value the user did not choose

It even clips requested bounds to the validated model domain **and appends a note
saying it did**:

```python
notes.append(
    f"{name}: requested bounds [{lo:g}, {hi:g}] extend beyond "
    f"the validated model domain [...]; the search is clipped to the "
```

Then `raosim/mdo/execute.py:347`:

```python
nlp = solve_min_mass(
    mission,
    _isp_floor(spec),
    objective=objective,
    allow_incomplete_physics=spec.models.allow_incomplete_physics,
    contour_provider=contour_provider,
)
```

None of `fixed`, `bounds`, `enumerations` or `active` is passed. And
`solve_min_mass` (`raosim/mdo/nlp.py:207`) rebuilds its own at `:232`:

```python
space = default_design_space(mission)
```

So an optimize job's declared design intent is computed carefully, recorded in the
resolved job, reported to the user — and has no effect on the search. The *analyze*
path is fine: it uses `job.design_vector()` (`job.py:598`).

### What stands in the way

You cannot fix this by adding a `design_layout=` argument, because the layout type
forbids the shape you need. `DesignLayout.__post_init__`
(`raosim/mdo/schema.py:82`):

```python
allowed = (_DESIGN_BASE_FIELDS, _DESIGN_CONTRACT_FIELDS)
if self.active_names not in allowed:
    raise ValueError(
        "DesignLayout active_names must be the ordered 10-variable "
        "hardware layout or that layout followed by OF"
    )
```

The active set is **all ten variables, or those ten plus O/F**. There is no
representation for "Pc pinned at 3 MPa, the other nine free". That rigidity is why
the resolver's four dicts had nowhere to go and ended up as report fields.

### The fix

**(a) Generalise `DesignLayout` to a reduced layout, keeping the serialization
contract invariant.** This is the discipline the existing docstring already states —
*"Serialized/state contracts always use `contract_names`, which has eleven physical
values"* — and it is exactly what makes the change safe:

- `active_names` becomes any ordered subset of the contract fields.
- Add `pinned: Mapping[str, float]`.
- Invariant, asserted in `__post_init__`: `set(active_names) | set(pinned)` equals the
  contract set, and the two are disjoint.
- `contract_names`, `of_is_variable` and the 11-value serialized vector are
  **unchanged**. Every state, snapshot and parity contract keeps its current shape.

The optimizer then searches `active_size` variables and reconstitutes the full
11-value contract vector by merging `pinned`. `raosim/mdo/engine.py:728-756` already
accepts either an active-layout vector or an 11-value contract vector, so it is
structured for this.

**(b) Thread it.** `solve_min_mass` takes `layout: DesignLayout` and **drops its
internal `default_design_space(mission)` call** — the layout arrives resolved or the
function does not run. The eight consumers of `design_layout()` (`execute.py:366`,
`engine.py:728`, `job.py:613`, `nlp.py:176`/`:263`, `postprocess.py:104`/`:695`/`:1017`)
each take the resolved layout instead of asking the mission for a fresh one.
`postprocess.py:104` already accepts an optional `layout` argument, so it is
half-done.

**(c) Non-Isp targets.** `_isp_floor` collapses the whole target set to one float.
Either compile every target into an optimizer constraint, or state explicitly in the
result that a given target is a **postsolve check** and not a search constraint. What
must not persist is a target that reads as enforced and is not.

**(d) Enumerations** expand into explicit separate solves, not a relaxed continuous
variable.

### The test that proves it

A pinned variable comes back bitwise (or within a stated tolerance) equal to the
request. Every returned free value satisfies the *requested* bounds, not just the
model-domain bounds. And a direct one: two jobs identical except for a pin produce
different `nlp.design` — which fails today.

**Cost: the largest of the four.** Contained to six modules, but it touches the
design-vector contract, so the parity and snapshot suites are the real gate.

---

## 3. W03 — architecture, schema and output contracts accept without implementing

Three separate holes, same mechanism (A): parsed, never read.

### 3.1 Architecture

`ArchitectureSelection` (`raosim/mdo/job.py:335`) carries `propellant`, `feed`,
`injector`, `cooling`, `liner_material`, `closeout_material`.
`resolve_engine_job` reads **`propellant` and the two materials**. `feed`, `injector`
and `cooling` are never read by anything.

So `feed: pressure_fed, injector: coaxial, cooling: ablative` is accepted and
resolved with the ordinary electric-pump / pintle / regen design variables, and the
report does not say a request was ignored. The user gets a pump-fed regen engine
under a pressure-fed ablative heading.

### 3.2 Schema version

`_build_spec` (`lrekit/job_cli.py:73`) lists `schema_version` among the permitted
document keys — so an unknown version passes the unknown-key check — and then
`:121-130` constructs `EngineJobSpec` **without passing it**. The field takes its
default (`raosim/mdo/job.py:514`) and is serialized back out at `:642`. A document
declaring `UNSUPPORTED_V999` is echoed as the current version.

That is worse than ignoring it: the output *asserts* a version the input did not
claim.

### 3.3 Output policy

`OutputPolicy` (`raosim/mdo/job.py:487`) declares `parts`, `cad_formats`,
`cad_fidelity`, `reports`, `output_dir`, `allow_partial_package`. It is constructed
at `lrekit/job_cli.py:130` and **consumed nowhere** — a repository-wide search finds
no other reference to `output_policy`, `cad_fidelity` or `allow_partial_package`.

The path confirms it: `job_cli.py:196` calls `execute_engine_job(spec)`, whose
default is `size_pumps=False` (`execute.py:256`); its `reevaluate()` call passes no
output directory and no CAD settings (`execute.py:404`); the bridge defaults to
`output_dir=None, cad='none'` (`postprocess.py:158`). `--out FILE` writes
`EngineJobResult.to_dict()` — a compact result, not a part package.

**A job's output request does not drive artifact generation at all.**

### The fix — one capability registry, not three patches

All three are the same missing thing: **nowhere to look up whether a requested
option is implemented.** Patching each with a hard-coded `if feed != "electric_pump":
raise` produces three sets that rot independently.

Build a capability registry as a sibling of `MODEL_REGISTRY`, in the same shape as
`ModelProvenance` (`raosim/model_registry.py:17`) — the repository already has a
working, audited registry idiom with 61 entries and should not grow a second style:

```
capability_id, kind, status, implemented_by, requires, source, validity, notes
```

covering feed architectures, injector types, cooling schemes, CAD formats, CAD
fidelities, report kinds, and schema versions. Then:

- **Validation happens before solve.** Every architecture and output field is looked
  up; an unimplemented one is refused by name with its registry status, not resolved
  to a default.
- **`schema_version` is forwarded** and validated against the registry's supported
  set. An unknown version is refused.
- **`OutputPolicy` gets a consumer.** `execute_engine_job` takes the policy and
  drives `size_pumps`, the output directory, CAD formats and fidelity from it. A
  requested report or package either appears with its manifest or returns an explicit
  stage failure. `allow_partial_package` governs actual publication.
- **Requested CAD fidelity above what the registry supports is refused**, rather than
  silently ignored.

This registry is also `lrekit catalog`'s data source and the future UI's enumeration
source — Track M1. Building it here means Track M is mostly wiring by the time it
arrives.

### The test that proves it

For every field of `ArchitectureSelection` and `OutputPolicy`: the value changes the
run, or the run is refused naming the capability. `schema_version` round-trips or is
refused. A requested part package either exists on disk or the result carries a stage
failure explaining why not.

**Cost: medium.** The registry is a day; wiring `OutputPolicy` through
`execute_engine_job` → `reevaluate` → `postprocess` is the bulk.

---

## 4. W04 — a fake STEP is labelled valid reference CAD

### What actually happens

`build_engine_parts` (`raosim/mdo/build.py:242`) takes **paths to already-produced
files**, not exporter results. Its entire evidence for a part is `path.is_file()`.
Then `:297`:

```python
achieved = min(
    spec.supported_fidelity,
    max(requested_fidelity, BuildFidelity.L2_VALID_REFERENCE_CAD),
)
```

Read the inner expression: `max(requested_fidelity, L2)` makes **L2 a floor**. For
any part whose `supported_fidelity` is L2 or better, a file that merely exists
achieves L2 — regardless of what is in it. A text file named `fake.step` containing
`this is not a STEP file` is recorded `status="built"`,
`achieved_fidelity=L2_VALID_REFERENCE_CAD`, with a SHA-256 of its bytes as the
evidence.

The function never parses the CAD, checks units, verifies a geometry identity,
measures a solid volume, or consumes an export audit. It hashes bytes and asserts a
fidelity.

And `:321`:

```python
achieved_overall = (
    min(r.achieved_fidelity for r in built) if built
    else BuildFidelity.L0_NUMERICAL_ONLY
)
```

The minimum is over **built parts only**, so a manifest containing one fake part and
nine missing ones reports overall L2. The accompanying note — *"the ceiling is set by
the least-complete required part"* — is not what the code computes. `complete=False`
does correctly expose the missing parts, so the manifest is not lying about
completeness; it is lying about fidelity.

### What stands in the way

Two things, and the second is the reason to sequence W04 after W03.

**(a) The exporters do not return what the builder would need.** The fix is to
consume a typed exporter result carrying validation outcome, geometry identity,
declared units, material, measured solid volume, artifact hashes, and a failure
reason. Those results do not exist as a type today — though the *ingredients* do, and
in good shape: `injector_coaxial_cad.build_coaxial_part_set` already builds each body
once, transforms it once at the neutral-file boundary, and **measures volume from the
exact export body**; `coaxial_geometry_id` already produces an identity;
`design.py:2645` already refuses on identity mismatch. So this is lifting an existing
pattern into a shared type, not inventing one.

**(b) Nothing calls it.** A repository-wide search finds no caller of
`build_engine_parts` or `write_engine_build` outside `tests/test_mdo_build.py`.
**Fixing the fidelity logic changes nothing observable until the builder is wired into
a real run — and wiring it is exactly the `OutputPolicy` work in W03.** That coupling
is the single most important sequencing fact in this document.

### The fix

1. **Define `PartExportResult`** — the typed thing an exporter returns: valid/invalid,
   solid count, measured volume in declared units, geometry identity, material,
   artifact path and hash, and a failure reason when invalid.
2. **`build_engine_parts` consumes results, not paths.** Delete the path-string
   entry point so the weak call cannot be made.
3. **Fidelity requires positive evidence.** Each level states what must be proven;
   absent that evidence the part is `unavailable` with a reason, never floored to L2.
   Remove the `max(requested, L2)` floor entirely — requested fidelity is a *request*,
   and must never appear on the evidence side of the calculation.
4. **Compute readiness over applicable required parts**, not built ones. Report
   separately the best and worst fidelity actually achieved.
5. **Generate the expected part list from the architecture and test scope.** A fixed
   whole-engine electric-pump list cannot be the only contract for a component bench
   article or a pressure-fed design — and once W03's registry exists, the architecture
   is known, so the list can be derived rather than hard-coded.
6. **Distinguish manufactured parts, purchased components and reference envelopes.**
   They cannot share a fidelity ladder.

### The test that proves it

The audit's own reproduction, inverted: a text file named `fake.step` is recorded
`unavailable` with a parse-failure reason, and **cannot** reach L2. A genuine export
reaches L2 only when its result carries a measured volume and a matching geometry
identity. A manifest with one valid part and nine missing reports overall readiness
against the required set, not against the one part that exists.

**Cost: medium**, and mostly in the exporters rather than in `build.py`.

---

## 5. What stands in the way, collected

| # | Blocker | Blocks | Nature |
|---|---|---|---|
| B1 | `DesignLayout` cannot express a reduced active set (`schema.py:82`) | W02 | Type-level. Must be generalised before any pin can reach the optimizer. |
| B2 | No capability registry — nowhere to ask "is this implemented?" | W03 (all three parts) | Missing infrastructure. Patching without it produces three rotting hard-coded sets. |
| B3 | Exporters return paths, not typed results | W04 | Requires touching `regen_cad`, `injector_coaxial_cad`, `pump_cad_brep`, `engine_cad`. The pattern already exists in the injector; it needs lifting. |
| B4 | `build_engine_parts` has **no production caller** | W04 | Fixing it is inert until the `OutputPolicy` wiring from W03 exists. **W03 must land before W04 is observable.** |
| B5 | The mission override dict is hand-assembled (`job.py:672`) | W01, and any future mission field | Not a blocker so much as the mechanism that will reintroduce W01. Replace with a total mapping plus a domain assertion. |

None of the five is a physics problem. All five are contract-shape problems, which is
why this whole batch sits at stage 0–1 rather than competing with the geometry work.

---

## 6. Order

| Step | Work | Why here |
|---|---|---|
| **1** | **W01** — total mission mapping + altitude via `atmosphere.pressure` | Smallest, no blocker, and the mapping assertion is what stops the class of defect recurring. |
| **2** | **B2 → W03 architecture and schema version** | The registry is the prerequisite for two of the remaining three. Refusing an unsupported architecture is immediately valuable and independent of everything else. |
| **3** | **B1 → W02** | The largest single change. Do it after the registry so the enumeration expansion has somewhere to look up discrete choices, and while the parity suite is the only thing at risk. |
| **4** | **W03 output policy wiring** | Turns `OutputPolicy` from a dataclass into behaviour, and creates the production caller that W04 needs. |
| **5** | **B3 → W04** | Now observable. The fidelity logic and the exporter results land together, because either alone proves nothing. |

Two notes on the order. **W02 is placed third, not first**, despite being the most
serious in principle — because it is the only one whose fix can break a green test
suite, and it is worth having the registry and the mission mapping settled first so a
parity regression has an unambiguous cause. And **W04 is last by necessity, not
priority** — B4 means it cannot be demonstrated before step 4 lands.

---

## 7. The property these four share, and the gate that keeps them closed

Each fix above ends the same way: a **totality assertion**. Every field of
`MissionRequest` maps somewhere. Every field of `ArchitectureSelection` and
`OutputPolicy` is looked up in the registry. Every contract variable is either active
or pinned. Every fidelity claim names the evidence that supports it.

Adopt that as the standing gate for the job contract:

> **No field may be accepted that does not change the run, compile to a target or
> evidence requirement, or produce an explicit refusal. Adding a field without
> declaring which of the four it is must fail the test suite.**

That is a stronger statement than "fix these four", and it is the only version that
survives the next feature. It is also the input-side half of Track K0's rule — every
artifact stamps one identity, and assembly refuses a mismatch. Same discipline, both
ends of the pipeline.
