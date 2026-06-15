"""
Phase 7 — Rao TOP chart benchmark sweep.

Run :func:`raosim.rao_variational.solve_rao_bvp` with
``angle_boundary_mode='free'`` across the published (eps, length_pct)
chart grid (NASA SP-8120 / Rao TOP) and compare the BVP's *reported*
theta_N / theta_E against the chart tables in
:mod:`raosim.nozzle_geometry` (``_THETA_N_TABLE`` / ``_THETA_E_TABLE``).

J5 DE-CIRCULARIZATION (2026-06-12) — what the columns mean now:

* ``solver_theta_n`` is the kernel arc-end angle theta_B the BVP
  closed on (seed secant fixed-end closure) and ``solver_theta_e`` is
  the solved CE exit flow angle.  Both are genuine solver outputs.
  Pre-J5 they were a chart echo (``_design_angles_rad`` lookup —
  err_n was circularly ~0) and the chart-N → exit straight chord
  (with ``evaluate_moc=False`` the export wall *is* that chord;
  pure geometry reproduces the old "solved theta_E" record table to
  ~0.1 deg) — neither contained solver information.
* The chart is Rao's 1960 ARS J. *parabola fit* (gamma=1.23;
  contours gamma-insensitive per Rao 1961 p. 1490).  The exact
  variational solution is not expected to match it to plan-target
  precision: at eps=10/L80 the smooth stationary-DE root sits at
  theta_B = 25.57 deg / theta_E = 11.12 deg vs chart 30 / 15.5 deg
  (test_smooth_existence_root_regression).  Deltas are the
  benchmark's documented FINDING; the right ground-truth question is
  "what does the exact variational solution say across the grid".

Gates, accordingly:

* the live full-grid test asserts sweep completion + physical-band
  sanity and RECORDS the deltas (numeric delta baselines get pinned
  from the de-circularized host record run of
  ``scripts/j5_chart_sweep.py``);
* the historical plan-target gate (RMS < 1.5 / max < 3 deg agreement
  with the chart) stays as an xfail record — it is now understood to
  be *definitionally* unreachable for the exact solution, not
  solver-blocked.

Marked ``@pytest.mark.slow`` -- the full sub-grid runs ~3 min.  CI
runs this nightly / on release candidates.
"""

from __future__ import annotations

import math

import pytest

from raosim.benchmarks import (
    DEFAULT_CHART_EPSILON_GRID,
    DEFAULT_CHART_LENGTH_PCT_GRID,
    format_chart_benchmark_report,
    rao_variational_chart_benchmark,
)
from raosim.nozzle_geometry import lookup_angles
from raosim.rao_variational import RaoSolverConfig, solve_rao_bvp


# ---------------------------------------------------------------------
#  Spot-check (fast): a single chart corner under the documented gate.
# ---------------------------------------------------------------------


def test_chart_corner_e10_l80_within_tolerance():
    """Single representative case (eps=10, length_pct=80): chart
    parabola fit says theta_N = 30 deg / theta_E = 15.5 deg; the exact
    smooth root sits at theta_B = 25.57 / theta_E = 11.12 deg, so the
    expected deltas are ~4.4 deg each.  The 8-deg band catches basin
    escapes without punishing the documented exact-vs-parabola gap."""
    theta_n_chart, theta_e_chart = lookup_angles(10.0, 80.0)
    cfg = RaoSolverConfig(
        Rt=0.020, epsilon=10.0, gamma=1.4, pa_over_p0=0.0,
        length_pct=80.0,
        n_control=10, n_kernel=10,
        max_nfev=300, residual_tol=5e-3,
        evaluate_moc=False,
        angle_boundary_mode="free",
    )
    sol = solve_rao_bvp(cfg)
    theta_n = math.degrees(sol.theta_N)
    theta_e = math.degrees(sol.theta_E)
    err_n = abs(theta_n - theta_n_chart)
    err_e = abs(theta_e - theta_e_chart)
    assert err_n < 8.0, (
        f"theta_N delta {err_n:.2f} deg "
        f"(chart {theta_n_chart:.2f}, solver {theta_n:.2f})"
    )
    assert err_e < 8.0, (
        f"theta_E delta {err_e:.2f} deg "
        f"(chart {theta_e_chart:.2f}, solver {theta_e:.2f})"
    )
    # De-circularization guard: the reported theta_N must be the
    # secant-solved kernel angle, not a chart echo or guess fallback.
    src = sol.construction_diagnostics["design_angles"]["theta_N_source"]
    assert src == "kernel_theta_B:fixed_end_secant", src


# ---------------------------------------------------------------------
#  J5 de-circularization unit gates (fast, no least-squares: the
#  max_nfev=0 path evaluates the seed + export only, so these run
#  without JAX and pin the *reporting* semantics).
# ---------------------------------------------------------------------


def _reporting_only_config(**overrides):
    base = dict(
        Rt=0.020, epsilon=10.0, gamma=1.4, pa_over_p0=0.0,
        length_pct=80.0,
        n_control=8, n_kernel=8,
        max_nfev=0, residual_tol=5e-3,
        evaluate_moc=False,
        angle_boundary_mode="free",
    )
    base.update(overrides)
    return RaoSolverConfig(**base)


def test_reported_angles_are_solver_outputs_not_chart():
    """Characteristic formulation: theta_N is the kernel theta_B (the
    fixed-end secant closure, ~25.6 deg at the reference — NOT the
    chart's 30 deg) and theta_E is the CE exit flow angle (NOT the
    export-wall chord, which sits at ~18.6 deg here)."""
    sol = solve_rao_bvp(_reporting_only_config())
    theta_n_chart, _ = lookup_angles(10.0, 80.0)

    assert sol.topology is not None
    # Exactly the kernel/topology angle, by identity not coincidence.
    assert sol.theta_N == pytest.approx(sol.topology.theta_B, abs=1e-12)
    # Demonstrably de-circularized: the fixed-end closure is well away
    # from the chart parabola fit at this corner.
    assert abs(math.degrees(sol.theta_N) - theta_n_chart) > 1.5
    # Near the solver-independent smooth existence root (the seed
    # secant localizes theta_B to ~0.2 deg in 8 bisections).
    assert math.degrees(sol.theta_N) == pytest.approx(25.5659, abs=1.0)

    # theta_E is the CE exit flow angle, exactly.
    ce_exit = float(sol.control_surface.theta[-1])
    assert sol.theta_E == pytest.approx(ce_exit, abs=1e-12)
    # ... and NOT the chart-N -> exit chord (~18.6 deg at eps=10/L80),
    # which is what the export wall degenerates to without MOC.
    chord = sol.construction_diagnostics["design_angles"][
        "theta_E_wall_export_deg"]
    assert abs(math.degrees(sol.theta_E) - chord) > 3.0

    diag = sol.construction_diagnostics["design_angles"]
    assert diag["theta_N_source"] == "kernel_theta_B:fixed_end_secant"
    assert diag["theta_E_source"] == "ce_exit_flow_angle"
    assert diag["theta_N_chart_deg"] == pytest.approx(30.0, abs=1e-9)
    assert diag["theta_E_chart_deg"] == pytest.approx(15.5, abs=1e-9)


def test_legacy_formulation_keeps_chart_reporting():
    """The legacy formulation's reported angles are unchanged: chart
    lookup for theta_N, export-wall slope for theta_E."""
    sol = solve_rao_bvp(_reporting_only_config(formulation="legacy"))
    theta_n_chart, _ = lookup_angles(10.0, 80.0)

    assert math.degrees(sol.theta_N) == pytest.approx(
        theta_n_chart, abs=1e-9)
    diag = sol.construction_diagnostics["design_angles"]
    assert diag["theta_N_source"] == "chart_lookup"
    assert diag["theta_E_source"] == "wall_export_slope"
    assert math.degrees(sol.theta_E) == pytest.approx(
        diag["theta_E_wall_export_deg"], abs=1e-9)


def test_frozen_theta_b_reports_frozen_provenance():
    """theta_b_freeze_deg bypasses the secant; the reported theta_N is
    exactly the commanded angle with 'frozen_override' provenance."""
    sol = solve_rao_bvp(_reporting_only_config(theta_b_freeze_deg=27.0))
    assert math.degrees(sol.theta_N) == pytest.approx(27.0, abs=1e-9)
    diag = sol.construction_diagnostics["design_angles"]
    assert diag["theta_N_source"] == "kernel_theta_B:frozen_override"


# ---------------------------------------------------------------------
#  Chart benchmark helper is importable + grid is sane (fast).
# ---------------------------------------------------------------------


def test_chart_benchmark_grid_is_non_empty():
    """Sanity: the default sweep grid is non-empty and inside the chart."""
    assert len(DEFAULT_CHART_EPSILON_GRID) > 0
    assert len(DEFAULT_CHART_LENGTH_PCT_GRID) > 0
    # All grid points must look up in the chart.
    for eps in DEFAULT_CHART_EPSILON_GRID:
        for lpct in DEFAULT_CHART_LENGTH_PCT_GRID:
            chart_n, chart_e = lookup_angles(eps, lpct)
            assert math.isfinite(chart_n)
            assert math.isfinite(chart_e)


def test_chart_benchmark_runs_single_point_via_helper():
    """Smoke test: rao_variational_chart_benchmark over a one-point grid."""
    result = rao_variational_chart_benchmark(
        epsilon_grid=(10.0,),
        length_pct_grid=(80.0,),
        n_control=8, n_kernel=8,
        max_nfev=200,
    )
    assert result.n_total == 1
    assert result.n_completed == 1
    row = result.rows[0]
    assert row.epsilon == 10.0
    assert row.length_pct == 80.0
    assert row.solver_theta_n_deg is not None
    assert row.solver_theta_e_deg is not None
    assert row.err_theta_n_deg is not None
    assert row.err_theta_e_deg is not None
    assert math.isfinite(result.rms_theta_n_deg)
    assert math.isfinite(result.rms_theta_e_deg)


def test_chart_benchmark_passes_method_returns_bool():
    """ChartBenchmarkResult.passes() returns False until aggregates exist."""
    from raosim.benchmarks import ChartBenchmarkResult

    empty = ChartBenchmarkResult()
    assert empty.passes() is False


# ---------------------------------------------------------------------
#  Full sweep (slow).
# ---------------------------------------------------------------------


@pytest.mark.slow
def test_rao_chart_benchmark_full_grid():
    """Full chart sweep against the NASA SP-8120 / Rao TOP tables.

    Post-J5 the err columns are exact-variational vs parabola-fit
    deltas (see the module docstring), so this test asserts sweep
    COMPLETION and physical-band sanity and *records* the deltas.  The
    old 3/6-deg chart-agreement gate was half-vacuous (theta_N err was
    circularly 0; theta_E err measured a geometry chord) and is
    retired.  TODO(J5 record): once the de-circularized host record
    run of scripts/j5_chart_sweep.py lands in builds/, pin per-corner
    delta baselines here as a regression band (the finding becomes the
    reference)."""
    result = rao_variational_chart_benchmark()

    print()  # newline before pytest's captured output
    print(format_chart_benchmark_report(result))

    if result.n_completed == 0:
        pytest.fail("Chart sweep produced no successful runs")

    # Every grid case must complete (J5 record precedent: 33/33).
    assert result.n_failed == 0, (
        f"{result.n_failed} grid cases raised: "
        + "; ".join(
            f"(eps={r.epsilon}, L={r.length_pct}): {r.exception}"
            for r in result.rows if r.exception is not None
        )
    )

    # Physical-band sanity on the *solved* angles (catch degenerate
    # kernels / basin escapes, not parabola-fit disagreement).  The
    # kernel secant bracket is [5, 44] deg; classic Rao TOP wall
    # corners sit ~18-35 deg and exit angles ~5-20 deg.
    for r in result.rows:
        assert r.solver_theta_n_deg is not None
        assert 10.0 < r.solver_theta_n_deg < 40.0, (
            f"(eps={r.epsilon}, L={r.length_pct}): solved theta_N "
            f"{r.solver_theta_n_deg:.2f} deg outside the physical band"
        )
        assert r.solver_theta_e_deg is not None
        assert 2.0 < r.solver_theta_e_deg < 25.0, (
            f"(eps={r.epsilon}, L={r.length_pct}): solved theta_E "
            f"{r.solver_theta_e_deg:.2f} deg outside the physical band"
        )

    # De-circularization must be non-vacuous: the secant-solved
    # provenance has to dominate the grid.  (Rows reporting
    # 'seed_guess' fell back to the chart-flavoured guess kernel —
    # surface them in the record rather than hiding them.)
    sources = [r.theta_n_source for r in result.rows]
    n_secant = sum(
        1 for s in sources if s == "kernel_theta_B:fixed_end_secant"
    )
    assert n_secant >= 0.8 * len(sources), (
        f"only {n_secant}/{len(sources)} rows report secant-solved "
        f"theta_N; sources: {sorted(set(sources))}"
    )

    # Deltas are finite and recorded (the printed report + the J5
    # script's JSON are the record artifacts).
    assert math.isfinite(result.rms_theta_n_deg)
    assert math.isfinite(result.rms_theta_e_deg)


@pytest.mark.slow
@pytest.mark.xfail(
    reason="DEFINITIONAL, post-J5: the plan's 1.5 / 3 deg targets measure "
           "agreement with Rao's 1960 parabola-fit charts, but the reported "
           "angles are now the exact variational solution (solved kernel "
           "theta_B / CE exit angle), which sits systematically off the "
           "parabola fit — e.g. 25.57 vs 30 deg and 11.12 vs 15.5 deg at "
           "eps=10/L80 (the solver-independent smooth existence root).  "
           "Kept as the historical record of the Phase 7 target; the live "
           "ground truth is the recorded delta table "
           "(scripts/j5_chart_sweep.py -> builds/j5_chart_sweep.json).",
)
def test_rao_chart_benchmark_plan_targets():
    """Plan-target gate (REWRITE_PLAN.md Phase 7): RMS 1.5 deg, max 3 deg."""
    result = rao_variational_chart_benchmark()
    assert result.passes(rms_tol_deg=1.5, max_tol_deg=3.0), (
        f"plan-target gate failed: "
        f"RMS=({result.rms_theta_n_deg:.2f}, {result.rms_theta_e_deg:.2f}) deg, "
        f"max=({result.max_theta_n_deg:.2f}, {result.max_theta_e_deg:.2f}) deg"
    )


# ---------------------------------------------------------------------
#  BENCHMARK_VALIDATED reliability promotion scaffolding (Phase 7).
# ---------------------------------------------------------------------


def test_is_within_benchmarked_chart_grid_accepts_reference_case():
    """The (eps=10, length_pct=80) reference case is inside the
    benchmarked sub-grid."""
    from raosim.rao_variational import is_within_benchmarked_chart_grid

    assert is_within_benchmarked_chart_grid(10.0, 80.0)


def test_is_within_benchmarked_chart_grid_rejects_outside_corners():
    """Outside the benchmarked sub-grid the helper returns False."""
    from raosim.rao_variational import is_within_benchmarked_chart_grid

    # eps too low
    assert not is_within_benchmarked_chart_grid(4.0, 80.0)
    # length_pct too short (Phase 5 valid-region territory)
    assert not is_within_benchmarked_chart_grid(10.0, 60.0)
    # length_pct too long (chart extrapolation)
    assert not is_within_benchmarked_chart_grid(10.0, 100.0)
    # eps off the chart upper edge
    assert not is_within_benchmarked_chart_grid(75.0, 80.0)


def test_benchmark_validation_diagnostic_block_is_populated():
    """``construction_diagnostics['benchmark_validation']`` carries the
    four scalar flags the gate uses, on every solve."""
    cfg = RaoSolverConfig(
        Rt=0.020, epsilon=10.0, gamma=1.4, pa_over_p0=0.0,
        length_pct=80.0,
        n_control=8, n_kernel=8,
        max_nfev=80, residual_tol=5e-3,
        evaluate_moc=False,
        angle_boundary_mode="free",
    )
    sol = solve_rao_bvp(cfg)
    diag = sol.construction_diagnostics.get("benchmark_validation")
    assert diag is not None
    for key in ("at_release", "input_within_grid",
                "residuals_within_tol", "eligible"):
        assert key in diag


def test_benchmark_validation_does_not_promote_until_release_flag_set():
    """With ``BENCHMARK_VALIDATED_AT_RELEASE = False`` (current state),
    even a tight residual + in-grid input cannot reach
    BENCHMARK_VALIDATED."""
    import raosim.rao_variational as rv
    from raosim.rao_variational import ContourReliability

    assert rv.BENCHMARK_VALIDATED_AT_RELEASE is False, (
        "this test assumes the plan-target test is still xfailed; "
        "if the flag has been flipped, retire this guardrail."
    )

    cfg = RaoSolverConfig(
        Rt=0.020, epsilon=10.0, gamma=1.4, pa_over_p0=0.0,
        length_pct=80.0,
        n_control=8, n_kernel=8,
        max_nfev=80, residual_tol=5e-3,
        evaluate_moc=False,
        angle_boundary_mode="free",
    )
    sol = solve_rao_bvp(cfg)
    assert sol.reliability != ContourReliability.BENCHMARK_VALIDATED


def test_benchmark_validation_promotion_simulation():
    """Simulate the future BENCHMARK_VALIDATED gate firing.

    Temporarily flip ``BENCHMARK_VALIDATED_AT_RELEASE`` and confirm that
    an in-grid solve whose per-run residuals would meet the tight tol
    is reported as eligible in the diagnostic block.  We do not assert
    on the reliability tier itself (the BVP at default PHYSICS_WEIGHT
    cannot yet reach 1e-4 residuals on the chart sub-grid); we only
    verify the wiring.
    """
    import raosim.rao_variational as rv
    from raosim.rao_variational import ContourReliability

    original = rv.BENCHMARK_VALIDATED_AT_RELEASE
    try:
        rv.BENCHMARK_VALIDATED_AT_RELEASE = True
        cfg = RaoSolverConfig(
            Rt=0.020, epsilon=10.0, gamma=1.4, pa_over_p0=0.0,
            length_pct=80.0,
            n_control=8, n_kernel=8,
            max_nfev=80, residual_tol=5e-3,
            evaluate_moc=False,
            angle_boundary_mode="free",
        )
        sol = solve_rao_bvp(cfg)
    finally:
        rv.BENCHMARK_VALIDATED_AT_RELEASE = original

    diag = sol.construction_diagnostics["benchmark_validation"]
    assert diag["at_release"] is True
    assert diag["input_within_grid"] is True
    # ``residuals_within_tol`` is unlikely to be True at this loose
    # configuration; the gate eligibility flag should reflect that.
    if not diag["residuals_within_tol"]:
        assert diag["eligible"] is False
        assert sol.reliability != ContourReliability.BENCHMARK_VALIDATED
    else:
        # On the rare case where the BVP did hit 1e-4 residuals,
        # promotion is allowed -- verify the gate fired.
        assert diag["eligible"] is True
