# Legacy Code Audit

This audit covers the NASA/JHU `MOC_Grid_BDE` code that is being ported into
`raosim.nasa_moc` and compared through `raosim.legacy_io`.

## Current Python Coverage

Implemented:

- `raosim.legacy_io` parses all files in `outputs_M3.5Perf`.
- `raosim.transonic_kernel` contains the Kliegel-Levine throat kernel.
- `raosim.nasa_moc` has `MOCNode`, `MOCKernel`, `RaoTopology`,
  `build_kernel`, `calc_mdot_bd`, `calc_lrc_de`, `find_point_e`, and
  `set_theta_b`.
- Tests compare the KL throat-plane wall Mach to NASA `wall.out`.

Known gaps:

- `build_kernel()` can still fall back to an arc-following BD surrogate when
  the selected TT' starting line is incompatible with the source-shaped row
  march. The corrected KL starting-line path remains in this diagnostic state;
  the source dtheta row march reaches the wall with the Sauer/Hall seed and
  the default mass gate.
- The Python row march does not yet reproduce NASA `wall.out`.
- `solve_rao_bvp()` still uses the NASA code mostly as a seed/topology helper.
  It is not yet the canonical Rao construction.
- `NASA_REFERENCE_MATCHED` exists as a reliability tier, but no production
  promotion path uses it yet.
- `calc_bde_region()` now ports the first source-shaped pass through
  `CalcBDERegion`, `CalcRemainingMesh`, and `CalcWallContour` for BFE
  comparison artifacts. This is an overlay diagnostic, not NASA reference
  parity.
- `build_source_contour_from_kernel()` now exposes an explicit visible-source
  contour artifact through kernel, D/DE, BFE, and wall extraction. It reports
  length closure and `CropNozzleToLength` as incomplete instead of folding that
  status into the legacy BVP wall export.

## Oracle Visibility

Added artifacts:

- `docs/nasa_jhu_file_list.txt`
- `docs/legacy_code_audit.md`
- `docs/rice_jhu_moc_topology_map.md`
- `docs/nasa_tt_prime_provenance.md`
- `scripts/compare_nasa_reference.py`

The comparison script reports current RMS differences for available Python
outputs and marks missing canonical outputs as unavailable with a reason. That
behavior is intentional: it keeps the current gaps visible instead of smearing
them into a misleading numeric tolerance.

`scripts/compare_nasa_reference.py --output-dir debug_outputs/nasa_comparison`
also writes persisted artifacts:

- `report.json`
- `metrics.csv`
- normalized station-diff CSVs for available wall/kernel comparisons

`report.json` also separates the two Phase 12 tracks:

- `canonical_reference_track`: `visible_source_port`;
- `comparison_track`: `historical_fixture_overlay`;
- `source_port_matched`: not evaluated by the fixture-overlay harness;
- `fixture_overlay_available`: true when the historical NASA/JHU files are
  present;
- `fixture_overlay_is_promotion_authority`: false;
- `fixture_generator_provenance`: `unresolved` for the M3.5Perf TT' fixture;
- `nasa_reference_matched_eligible`: false while source-port parity is not
  certified.

## Current First Numerical Blocker

The first mismatch is upstream of the full row march: Python TT' does not match
NASA/JHU `TT'.out`.

Observed for the M3.5Perf reference with `Rt = Rd = 1`, `gamma = 1.4`,
`n_kernel = 101`:

- wall point matches: NASA `M = 1.17779`, Python `M = 1.177794`
- axis point does not: NASA `x = 0.724356`, `M = 1.5`; Python
  `x ~= 0.679071`, `M ~= 1.267618`

This means downstream `TT'BF_Kernel.out`, `LastKernel.out`, and `wall.out`
comparisons are contaminated before `CalcRRCsAlongArc` starts. The focused
xfail is
`tests/test_nasa_kernel.py::test_python_tt_prime_matches_nasa_tt_prime_rms_1e3`.

Algorithm-port update: the dtheta-form `CalcRRCsAlongArc` unit processes now
include special arc-wall insertion, source-shaped interior points, source
axis closure, and NASA's `dxdr` mass-flow sign. With
`starting_line_method="sauer_modified"` on the M3.5Perf geometry, the row march
reaches `theta_B` without relaxing `mdot_tol`. With the corrected
`"kliegel_levine"` starting line, the exact source interior equations still go
subsonic on the first row; the fallback remains visible instead of being
hidden by a tuning heuristic.

Additional audit note: the checked-in fixture does not appear to be reproduced
by the currently ported typo-corrected KL evaluator, and it also is not matched
by the literal visible `KLThroat` typos alone. Treat the fixture/source
relationship as an orphaned sample-output provenance case: the public upstream
history uploads source and sample outputs separately, with no matching
executable or source variant preserved. See
`docs/nasa_tt_prime_provenance.md`.

## C++ Routines That Still Need Exact Porting

Kernel row march:

- `CalcRRCsAlongArc` (ported for source dtheta row march; corrected-KL
  starting line remains incompatible)
- `CalcArcWallPoint` (ported, including special wall point insertion)
- `CalcInteriorMeshPoints` (ported for the initial arc march)
- `CalcAxialMeshPoint` (ported)
- `CalcMassFlowAndThrustAlongMesh`

Post-kernel mesh:

- `CalcBDERegion` (wall-to-DE seed rows retained by `calc_bde_region`)
- `CalcRemainingMesh` (ported for the BFE overlay grid)
- `CalcWallContour` (ported for the BFE wall-contour overlay)

D/BD/DE construction:

- `CalcMdotBD`
- `CalcLRCDE`
- `FindPointE`
- `RungeKutta`
- `RungeKuttaFehlberg`
- `Deriv`

Outer Rao driver:

- `SetThetaB`
- `CropNozzleToLength`
- `CalcContouredNozzle`

## Reliability Gate Intent

`ContourReliability.NASA_REFERENCE_MATCHED` should only be assigned when all of
these are true:

- the visible `MOC_GridCalc_BDE.cpp` source-port path is certified
  (`source_port_matched == true`);
- the source-port reference workflow is complete and has metrics for the
  canonical TT'BF, BFE, wall, centerline, and Rao outputs;
- historical fixture overlay files may be present, but they are diagnostic only
  and are not promotion authority while TT' provenance is unresolved;
- no kernel fallback was used;
- source-port comparison RMS for wall `x`, `r`, `M`, `theta`, and `p` is below
  `1e-3` against the canonical visible-source reference output;
- topology has B, BD, D, DE, E, and a wall contour;
- no endpoint postprocessing is used to cheat closure.

Until those conditions are enforced by code and tests, the solver must remain
below this tier.
