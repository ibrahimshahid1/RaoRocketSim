# Input and execution integrity — September 9, 2026

This implementation addresses the request/solver/report boundary defects in the September 5 audit. It preserves the existing numerical models and the separate in-progress Rao changes. It does not establish hardware manufacturability or qualification.

| Audit item | Implemented behavior | Remaining scope |
|---|---|---|
| D01 / W01, W03: accepted inputs disappear | Mission altitude and single-point pressure/duration reach the mission and resolved inputs. Explicit conflicts are rejected. Mission Isp joins effective targets. Unsupported mass/envelope, architecture, schema, model and output requests are refused. Qualification duration remains an evidence requirement with unknown coverage. | Additional architectures, multipoint operation and evidence-backed qualification need implementations. |
| D02 / W02: optimization loses variable policies | Fixed values are substituted before every evaluation; caller bounds govern free coordinates. Results retain the complete physical vector. Enumeration expands into separate jobs through the API and CLI. All-fixed optimization evaluates once. | Most target metrics remain postsolve checks, explicitly labelled as such. Only the delivered-Isp lower bound is passed as an optimizer target constraint. |
| D26: roughness changes units across pipelines | Absolute hydraulic roughness has an explicit metre-valued field. Legacy relative roughness preserves its meaning and converts using the actual evaluated channel dimensions. Both paths receive consistent hydraulic roughness. | Hydraulic roughness is not surface Ra. Other thermal/property/discretization differences remain separate model discrepancies. |
| D03 / D04: output policy and package honesty | Unified jobs publish requested analysis reports in isolated run directories, with hashes and a completeness manifest. Failure retains already-resolved design inputs. Missing reports are explicit. CAD requests are rejected because this service does not yet generate those packages. | Legacy nozzle/pump export stages, their failure isolation, and a unified CAD producer service remain open. |
| W04: a filename earns CAD fidelity | The path-only build utility records files/hashes as unverified; they cannot earn analytical or CAD fidelity. Overall fidelity considers every required part. | A positive producer-validation handoff binding geometry validity, units, part identity and configuration identity is still needed. |
| Model/input disagreement disappears in a successful verdict | Exact input crosscheck failures block success. Signed model deltas, missing comparisons and their diagnostic tolerances are serialized. Unadjudicated differences make model consistency unknown. Required verdict gates cannot be omitted. | Quantity-specific physical acceptance policies, measurements, uncertainty and calibration are not supplied by numerical parity. |
| CLI/guided inconsistencies | Both use the job contract. The guided partial/complete modes collect pins; offered objectives come from the executable registry. Unknown flags, duplicate JSON/YAML keys, malformed sections and unsupported schema versions are rejected. Saved jobs restore explicit units and retain custom propellant definitions. | The legacy CLI and other UI surfaces have not all been replaced by this service. |

## Input semantics

- An omitted ambient condition defaults to sea level. A single operating point may supply an explicit pressure. An explicitly supplied mission condition must agree with that point; it is never silently overridden.
- An omitted burn duration may be supplied by the operating point; otherwise the existing 120 s default applies. Two explicitly different durations are rejected.
- Altitude conversion uses the repository's identified simplified ISA implementation within its declared 0–32 km range. This change adds no atmosphere correlation. Outside that range, use an explicit ambient pressure for the single design point.
- `qualification_duration` cannot be proven by a steady analysis. Its coverage row remains unknown even when the numerical solve succeeds.
- Fixed and enumerated values retain units through document save/replay. Optimizer bounds are intersected with the configured search domain, with a note when clipped. Search limits are not evidence of physical validity.
- Unit symbols are case-sensitive. Unsupported spellings such as `Mbar` cannot silently become `mbar`. Overflow during unit conversion is rejected.
- `derived` pump-speed intent is refused during resolution because no job-level derivation is implemented. Unsupported selections do not fall back to default models.

## Roughness migration

`MissionSpec.channel_hydraulic_roughness_m` is an absolute hydraulic roughness length. `MissionSpec.channel_roughness` retains the legacy dimensionless relative roughness convention. Supplying an absolute value together with nonzero legacy relative roughness is an error. Both must be finite and nonnegative.

Resolved-input schema **1.2.0** stores `thermal.channel_hydraulic_roughness_m`, the input basis and, where applicable, the legacy relative value. The old relative value is converted using the resolved design dimensions, not nominal mission dimensions. The traditional `CoolingSpec.channel_roughness` continues to mean metres. A schema-1.1.0 resolved record must be re-resolved; changing its version label alone is not migration.

The regression exercises nonzero roughness at three channel widths through the actual MDO march and traditional cooling path, plus a varying traditional channel profile. It also checks smooth-zero compatibility and differentiability with respect to channel size. It does not assert that different thermal/property models produce identical overall results.

## Report contract

For example, add this section to an otherwise valid saved job:

```yaml
output_policy:
  reports: [summary, resolved_inputs, snapshot, comparison, coverage]
  output_dir: runs/integrity
  allow_partial_package: true
```

Run it through `lrekit validate JOB.yaml`, then `lrekit run JOB.yaml --out result.json`.

Each execution creates a unique directory containing `job.json`, `manifest.json`, the mandatory `summary.json` and the requested reports. The manifest starts as `in_progress`, records file hashes as reports are written, and ends as `complete` or `partial`. A complete report package can describe a failed solve: package completeness and scientific verdict are independent. Existing run directories are not reused.

Unavailable requested reports contain an explicit reason and are marked unavailable in the manifest. With `allow_partial_package: false`, missing requested reports also fail publication. With it enabled, publication may retain the partial package; it does not turn an unknown or failed analysis into a pass. Later-stage failure preserves solved design/input information when available. Interrupted runs retain their incomplete marker; automatic resume is not implemented.

Saved job replay assumes the same code, catalog and external model artifacts. This work preserves request intent and input identity; the complete resolved-configuration graph with all model/geometry/evidence hashes remains D05.

## Verification

**281 focused tests passed:** 270 across units, job resolution, CLI/guided flows, custom-catalog replay, optimizer policies, cooling/roughness, resolved inputs, execution verdicts and snapshots; 11 build-manifest checks. These are software regressions, not experimental validation or a full-repository test result.

A [real CLI report-output check](audit_work/INPUT_EXECUTION_SMOKE_2026-09-09.json) also verified all five report hashes, the altitude/duration handoff, a passing input crosscheck and explicitly unknown qualification/model coverage. The complete report package preserves the candidate's actual verdict rather than treating report generation as design success.

The 270-test command used `.venv-jax/bin/python -m pytest` with `test_units`, `test_mdo_job`, `test_lrekit_guided`, `test_lrekit_job_cli`, `test_job_execution_integrity`, `test_mdo_custom_propellant`, `test_mdo_variable_policies`, `test_mdo_nlp`, `test_mdo_schema`, `test_mdo_roughness_contract`, `test_mdo_cooling`, `test_mdo_cooling_gate`, `test_mdo_resolved_inputs`, `test_mdo_execute`, and `test_mdo_snapshot` under `tests/`. `tests/test_mdo_build.py` passed separately after the path-only fidelity correction.

## Next work

The remaining first-milestone work is the legacy pump sizing/export failure boundary and producer-validated artifact orchestration. The reported small-thrust sizing exception, large-thrust split-casing export failure, undeclared component feasibility domains and NPSH availability still need their dedicated handling. This patch does not claim those failures are repaired. Follow that with D05 configuration identity and the measurement/comparison/calibration loop; physical model adjustments need declared evidence and provenance.
