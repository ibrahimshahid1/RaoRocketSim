"""J5 record run: the Rao chart sweep under the JAX defaults.

DE-CIRCULARIZED (2026-06-12): the solver columns are now genuine
solver outputs — theta_N is the kernel arc-end angle theta_B the BVP
closed on (per-row ``theta_n_source`` records the provenance) and
theta_E is the solved CE exit flow angle.  The pre-J5 record run
(2026-06-11) measured a chart echo (err_n circularly ~0) and the
chart-N → exit straight chord (the entire old "solved theta_E"
signature — ~21.0° @L70 / ~18.5° @L80 / ~16.6° @L90,
epsilon-independent — is reproduced to ~0.1 deg by that chord's pure
geometry; its "low-ε theta_E gap" was chart-vs-chord, not physics).

What this record now documents: the systematic deltas between the
EXACT variational solution and Rao's 1960 ARS J. parabola-fit charts
(gamma=1.23; contours gamma-insensitive per Rao 1961 p. 1490).
Expected reference-point deltas: theta_B 25.57 vs chart 30 deg,
theta_E 11.12 vs chart 15.5 deg (the solver-independent smooth
existence root).  The ``*_gate`` fields are kept for the historical
record; post-J5 they are findings, not pass/fail criteria — the
plan-target xfail is definitional (see
tests/test_rao_chart_benchmark.py).

This is the *record* run for the J5 milestone: the suite's sweep at
the J4-gate physics weight (1.0); prints the per-case table and dumps
JSON for the repo record (used to pin the delta-baseline regression
test — see the full-grid test's TODO).

Run:  PYTHONPATH=. python scripts/j5_chart_sweep.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import raosim.rao_variational as rv
from raosim.benchmarks import (format_chart_benchmark_report,
                               rao_variational_chart_benchmark)

OUT = Path("builds/j5_chart_sweep.json")


def main() -> int:
    rv.PHYSICS_WEIGHT = 1.0
    result = rao_variational_chart_benchmark(progress=True)
    print(format_chart_benchmark_report(result))

    def _clean(v):
        if isinstance(v, float) and math.isnan(v):
            return None
        return v

    rows = [
        {k: _clean(v) for k, v in vars(row).items()} for row in result.rows
    ]

    source_counts: dict[str, int] = {}
    for row in result.rows:
        key = str(row.theta_n_source)
        source_counts[key] = source_counts.get(key, 0) + 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "physics_weight": 1.0,
        "de_circularized": True,
        "columns_note": (
            "solver theta_N = solved kernel theta_B; solver theta_E = "
            "solved CE exit flow angle; err_* = exact-variational vs "
            "Rao-1960 parabola-fit chart delta (a finding, not an error)"
        ),
        "theta_n_source_counts": source_counts,
        "n_completed": result.n_completed,
        "rms_theta_n_deg": _clean(result.rms_theta_n_deg),
        "rms_theta_e_deg": _clean(result.rms_theta_e_deg),
        "max_theta_n_deg": _clean(result.max_theta_n_deg),
        "max_theta_e_deg": _clean(result.max_theta_e_deg),
        "loose_gate_3_6": bool(result.passes(rms_tol_deg=3.0,
                                             max_tol_deg=6.0)),
        "plan_gate_1p5_3": bool(result.passes(rms_tol_deg=1.5,
                                              max_tol_deg=3.0)),
        "rows": rows,
    }, indent=2))
    print(f"wrote {OUT}")
    print("theta_N source counts:", source_counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
