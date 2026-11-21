from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from raosim.nasa_moc import set_theta_b
from raosim.rao_variational import _target_length, solve_rao_bvp, compare_rao_solutions_common_coordinates
from scripts.audit_moc_reference_profile import profile_config, summarize

OUT = Path(__file__).resolve().parent / "t16_seed_accuracy"
OUT.mkdir(exist_ok=True)
report = {"scope": "diagnostic seed accuracy using existing APIs; no default or acceptance changes", "levels": {}}
solutions = []


def bank():
    (OUT / "progress.json").write_text(json.dumps(report, indent=2) + "\n")


for label, config in [
    ("fine", profile_config(fine=True)),
    ("finest", replace(profile_config(fine=True), n_control=32, n_kernel=48,
                       kernel_dtheta_limit_deg=0.0625, kernel_max_rrcs=1000)),
]:
    trials = []
    selected = None
    report["levels"][label] = {"config": asdict(config), "seed_trials": trials}
    for budget in (8, 25, 40):
        if budget == 40 and trials[-1]["absolute_relative_exit_error"] <= 1e-8:
            break
        start = time.monotonic()
        topology, kernel = set_theta_b(
            config.Rt, config.epsilon, config.length_pct, config.gamma,
            config.pa_over_p0, theta_b_init_deg=config.thetaN_guess_deg,
            n_kernel=config.n_kernel, n_de_points=max(config.n_control, 12),
            starting_line_method=config.starting_line_method,
            L_target=_target_length(config.Rt, config.epsilon, config.length_pct),
            Ru=config.throat_upstream_radius_factor * config.Rt,
            Rd=config.throat_downstream_radius_factor * config.Rt,
            dtheta_limit=config.kernel_dtheta_limit_rad,
            mdot_tol=config.kernel_massflow_rel_tol,
            max_rrcs=config.kernel_max_rrcs, max_iter=budget, abs_tol=1e-8,
        )
        target = _target_length(config.Rt, config.epsilon, config.length_pct)
        trial = {
            "max_iter": budget, "requested_abs_relative_tolerance": 1e-8,
            "elapsed_s": time.monotonic() - start,
            "theta_B_deg": math.degrees(kernel.theta_B),
            "kernel_reached_wall": kernel.reached_wall,
            "kernel_rows": len(kernel.rrcs),
            "seed_exit_x_m": float(topology.E.x), "target_x_m": target,
            "absolute_relative_exit_error": abs(float(topology.E.x) / target - 1),
        }
        trials.append(trial)
        selected = (topology, kernel)
        bank()
        print(json.dumps({"label": label, "seed": trial}), flush=True)
    derived = replace(config, theta_b_freeze_deg=math.degrees(selected[1].theta_B))
    start = time.monotonic()
    solution = solve_rao_bvp(derived)
    solutions.append(solution)
    report["levels"][label]["solution"] = summarize(derived, solution)
    report["levels"][label]["solve_elapsed_s"] = time.monotonic() - start
    bank()
    print(json.dumps({"label": label, "promotion": solution.construction_diagnostics["gate_report"]["promotion"],
                      "endpoint_mismatch": solution.construction_diagnostics.get("endpoint_mismatch")}), flush=True)

report["refinement"] = compare_rao_solutions_common_coordinates(
    *solutions, Rt=1.0, gamma=1.23,
).to_dict()
report["complete"] = True
bank()
(OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"refinement_passes": report["refinement"]["passes"],
                  "metrics": report["refinement"]["metrics"]}), flush=True)
