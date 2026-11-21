from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from raosim.rao_variational import solve_rao_bvp, compare_rao_solutions_common_coordinates, _physical_bde_field_arrays
from scripts.audit_moc_reference_profile import profile_config, summarize, worst_field_differences, _finite_json

OUT = Path(__file__).resolve().parent / 't19_refinement'
OUT.mkdir(exist_ok=True)
config = replace(profile_config(fine=True, exact_exit=True, seed_max_iter=25, seed_abs_tol=1e-8),
                 n_control=32, n_kernel=48, kernel_dtheta_limit_deg=0.0625, kernel_max_rrcs=1000)
configs = [('baseline', config), ('refined', replace(config, n_control=48, n_kernel=64,
                                                    kernel_dtheta_limit_deg=0.03125, kernel_max_rrcs=2000))]
report = {
    'scope': 'one approved additional same-physics pair; no tolerance changes or automatic further refinement',
    'recomputed_baseline_reason': 'the prior native-script process ended without retaining full solution objects',
    'source_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
        'raosim/rao_variational.py', 'raosim/nasa_moc.py', 'raosim/transonic_kernel.py',
        'scripts/audit_moc_reference_profile.py', 'raosim/rao_residuals.py')},
}
solutions = []
start = time.monotonic()
try:
    for label, cfg in configs:
        began = time.monotonic()
        solution = solve_rao_bvp(cfg)
        solutions.append(solution)
        report[label] = summarize(cfg, solution)
        report[label]['elapsed_s'] = time.monotonic() - began
        pickle_path = OUT / (label + '_solution.pkl')
        with pickle_path.open('wb') as handle:
            pickle.dump(solution, handle, protocol=pickle.HIGHEST_PROTOCOL)
        xy, mach, theta = _physical_bde_field_arrays(solution, Rt=cfg.Rt)
        np.savez_compressed(OUT / (label + '_field.npz'), xy_over_rt=xy, mach=mach,
                            theta_rad=theta, wall_raw_m=solution.wall_raw,
                            ce_r_m=solution.control_surface.r, ce_M=solution.control_surface.M,
                            ce_theta_rad=solution.control_surface.theta, ce_x_m=solution.control_surface.x)
        report[label]['banked_solution'] = str(pickle_path)
        report[label]['banked_field'] = str(OUT / (label + '_field.npz'))
        (OUT / 'progress.json').write_text(json.dumps(_finite_json(report), indent=2, allow_nan=False) + '\n')
        print(json.dumps({'completed_level': label, 'elapsed_s': report[label]['elapsed_s'],
                          'promotion': report[label]['gate_report']['promotion']}), flush=True)
    evidence = compare_rao_solutions_common_coordinates(solutions[0], solutions[1], Rt=config.Rt, gamma=config.gamma)
    report['refinement'] = evidence.to_dict()
    report['worst_field_locations'] = worst_field_differences(solutions[0], solutions[1], Rt=config.Rt, gamma=config.gamma)
    code = 0 if all(solution.converged for solution in solutions) and evidence.applicable and evidence.passes else 2
except Exception as exc:
    report['exception'] = {'type': type(exc).__name__, 'message': str(exc),
                           'traceback': traceback.format_exc(), 'diagnostics': getattr(exc, 'diagnostics', None)}
    code = 1
report['exit_code'] = code
report['elapsed_s'] = time.monotonic() - start
report['artifact_sha256'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in OUT.glob('*') if path.suffix in ('.pkl', '.npz')}
(OUT / 'report.json').write_text(json.dumps(_finite_json(report), indent=2, allow_nan=False) + '\n')
print(json.dumps({'report': str(OUT / 'report.json'), 'exit_code': code, 'elapsed_s': report['elapsed_s']}), flush=True)
raise SystemExit(code)
