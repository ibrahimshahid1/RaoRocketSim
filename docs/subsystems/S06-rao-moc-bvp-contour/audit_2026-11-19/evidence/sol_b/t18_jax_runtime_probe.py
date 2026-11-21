from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from raosim.rao_variational import solve_rao_bvp
from scripts.audit_moc_reference_profile import profile_config, summarize, _finite_json

OUTPUT = Path(__file__).with_suffix('.json')
config = replace(profile_config(exact_exit=True), n_control=8, n_kernel=8,
                 solver_backend='jax', kernel_dtheta_limit_deg=0.5,
                 max_nfev=256, jax_constraint_weight_ladder=(1.0,))
start = time.monotonic()
try:
    solution = solve_rao_bvp(config)
    result = summarize(config, solution)
    result['runtime_completed'] = True
    result['scope'] = 'actual n8 static JAX exact-D+exact-exit integration; no mesh or full-physics acceptance claim'
    code = 0
except Exception as exc:
    result = {'runtime_completed': False, 'exception': str(exc), 'exception_type': type(exc).__name__,
              'traceback': traceback.format_exc(), 'diagnostics': getattr(exc, 'diagnostics', None)}
    code = 1
result['exit_code'] = code
result['elapsed_s'] = time.monotonic() - start
result['source_sha256'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
    'raosim/rao_variational.py', 'raosim/nasa_moc.py', 'raosim/jax/api.py', 'raosim/jax/assembly.py')}
OUTPUT.write_text(json.dumps(_finite_json(result), indent=2, allow_nan=False) + '\n')
print(json.dumps({'report': str(OUTPUT), 'exit_code': code, 'elapsed_s': result['elapsed_s']}))
raise SystemExit(code)
