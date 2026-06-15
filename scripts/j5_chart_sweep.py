"""J5 record run: the Rao chart sweep under the JAX defaults.

Post-flip (2026-06-11) the chart benchmark routes through the
characteristic formulation + JAX LM by default; the suite's
``test_rao_chart_benchmark_full_grid`` already enforces the loose
(3 deg RMS / 6 deg max) gate at PHYSICS_WEIGHT=0.05.  This script is
the *record* run for the J5 milestone: it executes the same sweep at
the J4-gate physics weight (1.0), prints the per-case table, and dumps
JSON for the repo record.

Honest-status caveat (carried from test_rao_chart_benchmark_full_grid's
docstring): theta_N is currently *reported from the chart lookup*
(``_design_angles_rad``), so its error column is circular (~0) — the
genuine BVP output is theta_E (integrated from the CE end state).
Closing the plan-target gate (RMS 1.5 / max 3 deg,
``test_rao_chart_benchmark_plan_targets`` xfail) requires reporting a
*solved* theta_N — the natural candidate post-12.4 is the converged
fixed-end topology theta_B, but reconciling the two chart conventions
in-repo (``nozzle_geometry._THETA_N_TABLE`` vs the benchmark tables)
against Rao 1958 / NASA SP-8120 must come first.  Tracked in the plan
STATUS block.

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

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "physics_weight": 1.0,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
