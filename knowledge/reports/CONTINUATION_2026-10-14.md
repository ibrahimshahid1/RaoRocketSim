# Scientific memory continuation

Historical snapshot: 2026-10-14. The [consolidated audit of 2026-10-16](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/knowledge/reports/KNOWLEDGE_GRAPH_AUDIT_2026-10-16.md) supersedes this report's counts, outstanding-work queue and interim validation. The inverter convention has since been corrected, all eighteen initially flagged damaged equations are superseded history, and explicit White derivation relationships have been banked. Corpus-wide mathematical completeness remains `NOT_ESTABLISHED`.

Baseline: the user's audit of commit `6c04d92` accurately described a functioning
auditing/retrieval layer with sparse scientific extraction. This continuation
adds source evidence and actual local artifact observations while preserving
that distinction.

| Evidence | Baseline | 2026-10-14 snapshot |
|---|---:|---:|
| Canonical publications | 248 | 248 |
| Publications with extracted scientific objects | 7 | 8 |
| Equations | 37 | 45 |
| Claims | 179 | 186 |
| Independently confirmed equations/claims | 0 | 14 |
| Models source-reviewed | 11/61 | 12/61 |
| Configuration snapshots | 0 | 17 |
| CAD references | 0 | 70 (64 current files observed) |
| Manufacturing reference plans | 0 | 15 |
| Constants classified UNKNOWN_BASIS | 223 | 223 |

## Completed evidence

The Lee electric-pump paper contributed eight equations, seven claims, three
assumptions, two applicability statements and eight case-specific inputs: 28
objects grounded to exact source-page text. Formula and table pages were also
visually checked. The source journal is
`workflows/results/electric-drive-2026-10-07.jsonl`.

At this snapshot, model `pump.electric_drive` was PARTIALLY_SUPPORTED. Source Eq. 29 defines inverter
mass using output-specific power, whereas `_drive_sizing` used input power with
the same density parameter. LREKit's separate drives, defaults and electrical
screens also extend beyond the paper. The then-existing Lee test checked motor and
battery mass, not inverter mass. This snapshot did not change the physics implementation.
The later [inverter remediation](/Users/ibrahimshahid/Documents/Github/RaoRocketSim/knowledge/reports/lee_inverter_remediation_2026-10-14.md)
corrected the NumPy and two MDO consumers and added inverter regressions; the
broader model remains partially supported.

An independent reviewer confirmed the earlier Rao extraction's nine equations
and five claims against newly rendered original PDF pages. Three necessary-
condition qualifications and the zero-ambient assumption were added. Verdicts,
source hashes, inspection evidence and scope are retained; replay is idempotent.
`workflows/results/rao1958-verification-2026-10-07.json` is the complete review.

The artifact harvester observes 17 existing reports: two serialized design inputs
and 15 injector-layout snapshots. It records current CAD hashes, report locators
and reproducible configuration payload selections. These are not 17 qualified
engine designs. Six references point outside this checkout and remain unavailable.
Manufacturing records are plans with no execution evidence. No measurements,
experiments, calibrations or comparison records were fabricated.

## Implementation and verification

Artifact observations participate in full rebuild, graph traversal, `artifacts`,
`artifact`, `sources` and task context. Changed/missing snapshots retain inactive
history and lose active graph associations. Verification rejects stale source
hashes even when a verdict omits its checksum, and replacement verdicts cannot
inherit a previous reviewer's evidence. A corrected mixed-PDF OCR gate handles
sparse pages individually and preserves explicit low confidence.
The native-text triage sampled 724 pages from all 248 canonical publications;
20 publications have sampled sparse pages requiring visual classification. Six
PDFs still held only in a local ZIP were restored with their original recorded
checksums. Archive origin metadata now survives into document/source retrieval;
missing or changed archive members cannot be reported as synchronized.

The repository `.venv` is absent; checks run using the existing `.venv-jax`.
The wrapper now selects an available repository environment. The knowledge suite,
full rebuild, hybrid task retrieval, artifact lookup and byte-identical artifact
replay were checked. Consult `latest_report.md` and `latest_validation.json` for
current counts and findings. Software checks do not constitute physical validation.

## Historical remaining-work queue and continuation

The snapshot had 240 canonical publications without structured scientific objects,
49 unchecked models (41 with local source-review routes and eight requiring
source routing or repository-basis review), 217 equations/claims awaiting independent
verification and 223 unknown-basis constants. These dated counts are not the current
verification/default inventory; the current report and coverage matrix own those counts.
The broad extraction gap remains, and a nearby citation still does
not supply a missing scientific basis.

The Lee inverter convention review and correction are now complete within the
bounded scope described above. The queued pump inducer NPSH (SP-8052) and
meanline/free-area relations (SP-8109) remain useful next source reviews. Use one bounded packet and
bank one completed source result before moving on. Test the local parser candidates
on a small fixed page set before corpus-wide replacement. Real measurement,
experiment and calibration ingestion requires actual datasets and metadata;
configuration/CAD observations alone cannot fill those gaps.
