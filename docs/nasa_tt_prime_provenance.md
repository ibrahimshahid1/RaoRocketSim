# NASA/JHU TT' Provenance

This note records the provenance check for
`MOC_Grid_BDE/outputs_M3.5Perf/TT'.out`.

## Conclusion

The exact executable or source variant that generated the checked-in
`TT'.out` is not present in the local NASA tree or in the public upstream
`nasa/Three-Dimensional-Nozzle-Design-Code` history.

Treat the M3.5Perf TT' values as sample output from an unpublished or
unpreserved pre-upload MOC_GRID_BDE run, not as output that is currently
reproducible from the visible `MOC_GridCalc_BDE.cpp` source.

The practical policy is now: the visible NASA/JHU source-port workflow is the
canonical reference track for `NASA_REFERENCE_MATCHED`; M3.5Perf is a
historical fixture overlay. Matching the orphaned TT' fixture is useful
diagnostic information, but it is not promotion authority.

## Upstream Timeline

Public upstream repository:
https://github.com/nasa/Three-Dimensional-Nozzle-Design-Code

Relevant upstream commits:

- `cf200d847614e0b84dd7f3dcc83db8f4cb6953ab`
  - Date: 2020-08-06 10:11:19 -0400
  - Message: `Initial upload of 2D MOC code`
  - Added `MOC_Grid_BDE/MOC_GridCalc_BDE.cpp`,
    `MOC_GridCalc_BDE_IO.cpp`, headers, and dialog code.
- `18756e93a6cfee0292f288b7bd46c74bf62ed9f9`
  - Date: 2020-08-06 10:53:43 -0400
  - Message: `Initial upload of Streamline Tracing Tool`
  - Added `STT2001/Example_M3.5Perf/MOC_Grid.plt` and
    `summary.out`. Those files already contain the current TT' throat line
    values, including wall `M = 1.17779` and axis `x = 0.724356`,
    `M = 1.5`.
- `22e0579d9d6f9e2560add95cf91a7c1749d19a15`
  - Date: 2020-08-06 12:04:30 -0400
  - Message: `Included example output for M3.5 perfect nozzle`
  - Added `MOC_Grid_BDE/outputs_M3.5Perf/TT'.out` and the rest of the
    MOC_GRID_BDE M3.5Perf output set.
- `5eb1c7474a1ba925825e65ed3b5071d5aed31dbc`
  - Date: 2020-08-06 12:06:56 -0400
  - Message: `Uploaded example outputs for Mach 3.5 perfect nozzle`
  - Re-uploaded STT M3.5Perf outputs under
    `STT2001/outputs_M3.5Perf`.

The MOC source has no upstream commits after `cf200d8` that could explain the
later sample output. The public repository also has no published releases or
checked-in executable.

## Online Search Check

An online search was repeated on 2026-06-09. The public NASA GitHub repository
and NASA Software Catalog both route to the same open-source Windows/MFC code:

- NASA Software Catalog entry `LEW-20180-1` lists the package as open source,
  Windows software and its `Download Now` link points to the GitHub repository:
  https://software.nasa.gov/software/LEW-20180-1
- The GitHub repository describes the three Windows/MFC programs and links the
  JHU/APL report as the mathematical/user reference:
  https://github.com/nasa/Three-Dimensional-Nozzle-Design-Code
- The GitHub repository has no published releases listed on the repository page.
- Searches for `MOC_GRID_BDE` executable/download variants, `TT'.out`,
  `TT'BF_Kernel.out`, `0.724356`, `CalcInitialThroatLine`, and `KLThroat`
  did not surface an alternate generator or source variant.

The NASA/JHU report remains high-level documentation for the TT' construction,
not a coefficient listing that can reconstruct the checked-in sample exactly.

## Local Evidence

Local source and fixture files match the public upstream checkout:

- `MOC_Grid_BDE/MOC_GridCalc_BDE.cpp` is identical to upstream `master`.
- `MOC_Grid_BDE/outputs_M3.5Perf/TT'.out` is identical to upstream `master`.
- No executable files are checked in under
  `Three-Dimensional-Nozzle-Design-Code-master`.

`STT2001/outputs_M3.5Perf/M3.5Perf.inp` is not a MOC_GRID_BDE run input. It is
an STT input that names already-generated files:

- `MOC_sl.plt`
- `MOC_Grid.plt`
- `summary.out`
- `friction_table.txt`

So STT preserves and consumes the MOC grid; it does not identify the MOC
executable/source variant that produced TT'.

## Numerical Evidence

For the M3.5Perf reference (`Rt = 1`, `Rd = 1`, `gamma = 1.4`,
`n_kernel = 101`), the visible source path and the checked-in fixture diverge
on the first line before the row march starts.

Current corrected Python KL port:

- wall: `x = 0`, `r = 1`, `M = 1.177794`
- axis: `x ~= 0.679071`, `M ~= 1.267618`

NASA/JHU fixture:

- wall: `x = 0`, `r = 1`, `M = 1.17779`
- axis: `x = 0.724356`, `M = 1.5`

The literal visible `KLThroat` typos were also simulated and do not reproduce
the fixture axis. The literal path lands around `x ~= 0.969666`,
`M ~= 1.427405`, so the mismatch is not explained by simply preserving the
visible transcription mistakes.

## External Primary Documentation

The NASA/JHU report describes TT' as the initial data line computed with the
modified Hall/Kliegel-Levine method, with an arbitrary Mach 1.5 cap that changes
the line shape when needed:

https://ntrs.nasa.gov/api/citations/20030067852/downloads/20030067852.pdf?attachment=true

That agrees with the high-level behavior of `CalcInitialThroatLine`, but it
does not provide a coefficient listing or source variant sufficient to recreate
the checked-in TT' values.

## Working Rule For The Port

Do not tune the Python KL coefficients blindly to hit `TT'.out`. The visible
source-port track is canonical. Keep the fixture mismatch visible through the
xfailed TT' parity test and the comparison artifacts in
`debug_outputs/nasa_comparison`, but treat those artifacts as overlays.

The comparison harness should be read as a historical fixture overlay, not as
a source-port certificate. Its persisted `report.json` records
`canonical_reference_track = "visible_source_port"`,
`comparison_track = "historical_fixture_overlay"`,
`source_port_matched = null`, `fixture_overlay_available = true`,
`fixture_overlay_is_promotion_authority = false`, and
`fixture_generator_provenance = "unresolved"` for this case.
