# TEST PROMPT — paste everything below this line into the CLI after making changes

---

Run the repository's standard regression check and evaluate whether my latest changes broke anything. Do not modify any source files during this task; this is evaluation only.

## Reference design under test

13 kN LOX/RP-1 engine, sea-level matched expansion (Pc 3.0 MPa, epsilon 4.524, Pe = Pa = 101325 Pa), non-scalar regen wall (station-wise GRCop-84 hot-wall + channel thermostructural sizing, Inconel 718 jacket), coaxial machined pintle injector, electric pumps, and the full co-runnable CAD/feature set (2026-07-13): wall STL + wall STEP with flange interface, regen coil STL + full-N regen B-rep STEP with manifolds, pintle machined STEP, pump STL + B-rep, engine assembly STEP, pump visualization. All parameters live in `examples/cli/test_13kn_sealevel_regen_allstl.args` — do not change them; they are literature-anchored (provenance in the file's header comments). Two companion variant checks cover the features that cannot co-run with this benchmark: `examples/cli/test_13kn_movable_pintle.args` (Son continuous-gap movable pintle; its five external-evidence release gates are EXPECTED to fail) and `examples/cli/test_13kn_rao_bvp_flowfield.args` (rao-bvp variational solve + flow field + Rao/BDE construction diagrams + crossing CSV; minutes of JAX runtime).

## Step 0 — preflight

Check `python -c "import cadquery"`. If missing, run `pip install cadquery` (needed for machined pintle STEP and pump B-rep STEP). If it cannot be installed, continue but mark the pintle/pump B-rep CAD checks SKIPPED-NO-CADQUERY in the report.

## Step 1 — benchmark pipeline check

```bash
PYTHONPATH=. python -m raosim @examples/cli/test_13kn_sealevel_regen_allstl.args
```

Runs the full benchmark contour (`--max-nfev 4000`) through the entire pipeline: thrust-sized throat, chamber, station-wise regen sizing, thermal, coaxial machined pintle injector, pumps, wall STL, pintle STEP, and pump STL+B-rep exports. This can take minutes. For a quick smoke run only, append `--max-nfev 0 --out builds/tests/13kn_sl_fast`. Capture stdout/stderr to a log. A nonzero exit code is an automatic FAIL: report the failing gate or traceback and stop.

## Step 2 — artifact completeness (all CAD present, all STL)

Under `builds/tests/13kn_sl_full/` (or `builds/tests/13kn_sl_fast/` when you intentionally overrode the output directory), verify every item exists and is non-empty:

- Nozzle/chamber/jacket: `wall.stl`, `jacket.stl`, `regen.stl`, `contour.csv`, `profile.png`, `regen_3d.png`, `summary.json`
- Wall B-rep + assembly (with CadQuery): `wall.step` (flange/bolt interface features), the full-N regen B-rep STEP with manifold bodies (`--regen-brep --regen-manifolds`), and the `--engine-assembly` engine STEP assembling wall/jacket/pintle/pump
- Pump visualization: `pump_particles.gif` (`--pump-visualize`)
- Injector: `pintle.json`, `pintle/pintle_parameters.json`, `pintle/pintle_dimensions.csv`, `pintle/injector_manufacturing_report.json`, and (with CadQuery) the coaxial machined STEP set: `pintle_body.step`, `pintle_tip.step`, `injector_body.step`, `orifice_plate.step`, `faceplate.step`, `injector_assembly_machined.step`. The report's `cad_export.flow_separation_audit.circuits_sealed` must be true.
- Pump (placeholder envelope meshes): `pump.json`, `pump_bom.json`, `pump/pump_parts/` with 11 STLs (fuel_/oxidizer_ impeller, inducer, diffuser_volute, motor, inverter; shared_battery_pack) plus `pump/pump_reference_assembly.stl`
- Pump (true B-rep geometry, requires CadQuery): `pump/pump_brep/` with per-part STEP + STL for each sized role — expect fuel_/oxidizer_ impeller, inducer, diffuser_ring, volute_casing, shaft (motor/inverter when sized), `<role>_pump.step` assemblies, and shared_battery_pack; `summary.json → pump_package.brep_diagnostics` must show `valid: true` and positive volume for every body. These B-rep parts (log-spiral camber blades, helical inducer) are the authoritative pump geometry; the pump_parts prisms are only envelope placeholders

Any missing required artifact is a FAIL (missing pintle/pump STEP with no CadQuery = SKIPPED, per Step 0).

## Step 3 — physics and gate assertions (from `summary.json` and the log)

1. `performance.thrust_N` == 13000 (throat is sized from target thrust; any drift is a FAIL)
2. `performance.Pe_pa` within 1% of `performance.Pa_pa` = 101325 (sea-level matched design)
3. `wall_sizing.feasible` == true, with thermal margin ≥ 1.05 and structural margin ≥ 1.0
4. `wall_geometry.stl_watertight` == true and `stl_boundary_edge_count` == 0
5. `wall_geometry.thickness_mode` == "station_wise_thermostructural_sizing" (this is the non-scalar regen requirement — FAIL if it reports a uniform reference wall)
6. Injector and pump gate blocks: 0 failing gates in the log ("N pass M warn 0 fail" for both)
7. Peak hot-gas wall T ≤ 1000 K (GRCop-84 catalog limit); coolant Δp within the 500 bar budget

Known, expected WARNs at this baseline (report them, but they are not regressions): RP-1 conservative 700 K coking screen exceeded (coolant-side wall; screen is non-gating, kerolox regen precedent noted in docs/regen_wall_model.md), and regen-vs-cycle fuel-flow closure ~2.5% (fail threshold 5%). A new warn that is not on this list should be reported as a CHANGE.

Reference snapshot from the verified seed smoke run (2026-07-02): Isp 254.2 s, Cf 1.441, c* 1774 m/s (delivered 1730), mdot 5.215 kg/s (O/F 2.27), Rt 30.94 mm, Me 2.78, sized t_hot 0.60 mm with 194 x 0.50 mm channels, peak wall 946 K, coolant dp ~5.6 bar, thermal margin 1.06, stress margin >= 1.7. Values in this ballpark on a fresh clone are healthy. Two config sensitivities verified 2026-07-13: the snapshot's Ru came from Cd 0.99 (now `--ru-factor 1.5`, Cd_Hall 0.98963 — sub-0.1% effect), and coolant dp is channel-height-sweep dependent — overriding `--channel-height-steps 6` down to 2 selects a shorter channel and reports ~13 bar with identical margins on both current and pre-drop code (config effect, not a hydraulics regression).

## Step 4 — baseline drift comparison

Baseline lives at `builds/tests/13kn_sl_baseline_summary.json`. If it does not exist, copy this run's `summary.json` there and say the baseline was initialized. Otherwise compare, and report any relative drift > 0.5% in: `Cf`, `Rt`, `performance.Isp_s`, `performance.c_star_effective_m_s`, `performance.mdot_total_kg_s`, `wall_sizing.t_hot_m`, `wall_sizing.channel_count`, cooling peak wall T, coolant Δp, and pump shaft power / battery mass from `pump.json`. Classify each drift as INTENDED (explained by my current change) or UNEXPLAINED (potential regression). Only overwrite the baseline if I explicitly say the new numbers are the intended new reference.

## Step 5 — deep solver check (optional; run when I say "full" or the change touches the solver/JAX/MOC path)

```bash
PYTHONPATH=. python -m raosim @examples/cli/test_13kn_sealevel_regen_allstl.args \
  --max-nfev 4000 --out builds/tests/13kn_sl_full
```

Expect minutes of runtime (JAX BVP solve). Assert it converges (no `--allow-unconverged` is passed, so non-convergence exits nonzero), then repeat Steps 2–4 against the full-solve output dir (keep a separate baseline `builds/tests/13kn_sl_full_baseline_summary.json`).

## Step 6 — report

End with a compact table: each check, PASS/FAIL/SKIPPED/CHANGE, the measured value vs expected, and a one-line verdict on whether my change is safe. If anything failed, point to the exact file/gate/log line and the most likely culprit in the code I changed — do not fix it unless I ask.
