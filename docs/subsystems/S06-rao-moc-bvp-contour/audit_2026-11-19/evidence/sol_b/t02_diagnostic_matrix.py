"""Diagnostic experiments only; does not change acceptance policies."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
import hashlib
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from raosim.rao_variational import (
    CHARACTERISTIC_RAO_RESIDUAL_BLOCKS,
    RaoSolverConfig,
    _link_sample,
    bde_mesh_links,
    solve_rao_bvp,
)
from raosim.rao_residuals import residual_Cminus_axisym, residual_Cplus_axisym

OUT = Path(__file__).resolve().parent
BASE = RaoSolverConfig(
    Rt=1.0, epsilon=4.973, gamma=1.23, pa_over_p0=0.0,
    length_pct=64.0452248702277, n_control=10, n_kernel=10,
    max_nfev=100, residual_tol=0.005, evaluate_moc=True,
    wall_method="bde", solver_backend="numpy",
)


def run_case(label, config):
    start = time.monotonic()
    solution = solve_rao_bvp(config)
    diag = solution.construction_diagnostics
    art = diag["bde_artifacts"]
    bfe = art["bde_region"]
    topo = art["nasa_topology"]
    mesh = bfe.rows or bfe.grid_rows
    links = bde_mesh_links(mesh, config.gamma)
    values = []
    de_coords = np.array([(p.x, p.r) for p in topo.DE])
    for link in links:
        p0, p1 = link.parent.to_flow_node(), link.child.to_flow_node()
        if p1.x < p0.x:
            p0, p1 = p1, p0
        func = residual_Cminus_axisym if link.family == "bde_row" else residual_Cplus_axisym
        value = func(p0, p1, config.gamma) / math.radians(1.)
        de_distance = [float(np.min(np.linalg.norm(de_coords - [p.x, p.r], axis=1))) for p in (p0, p1)]
        values.append(dict(residual_deg=float(value),de_node_distances=de_distance,**_link_sample(link)))
    worst = {}
    for family in ("bde_row", "bde_column"):
        sel = [v for v in values if v['family'] == family]
        worst[family] = sorted(sel,key=lambda v:abs(v['residual_deg']),reverse=True)[:5]
    result = dict(
        label=label,config=asdict(config),elapsed_s=time.monotonic()-start,
        reliability=solution.reliability.value,converged=solution.converged,
        Cf=solution.thrust_coefficient,residuals=asdict(solution.residuals),
        net_report=diag["net_report"],topology_closure=diag.get('topology_closure'),
        gate_report=diag.get("gate_report"),
        export_diagnostics={
            "export": diag.get("export"),
            "endpoint_mismatch": diag.get("endpoint_mismatch"),
            "raw_start_m": solution.wall_raw[0].tolist(),
            "raw_end_m": solution.wall_raw[-1].tolist(),
            "export_start_m": solution.wall_export[0].tolist(),
            "export_end_m": solution.wall_export[-1].tolist(),
            "target_x_m": float(solution.control_surface.length_target),
            "raw_exit_dx_m": float(solution.wall_raw[-1,0] - solution.control_surface.length_target),
        },
        source_sha256={
            name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ("raosim/rao_variational.py", "raosim/nasa_moc.py", "raosim/rao_residuals.py", "raosim/transonic_kernel.py", "raosim/jax/api.py", "raosim/jax/assembly.py")
        },
        optimizer={k:v for k,v in diag.items() if k.startswith('solver') or k in ('nfev','success','message')},
        warnings=solution.warnings,worst_edges=worst,
    )
    (OUT/f"{label}.json").write_text(json.dumps(result,indent=2,default=str)+"\n")
    report=result['net_report']
    print(json.dumps(dict(label=label,elapsed_s=result['elapsed_s'],reliability=result['reliability'],converged=solution.converged,max_scaled=solution.residuals.max_scaled,Cf=solution.thrust_coefficient,cplus_max=report.get('cplus_max'),cminus_max=report.get('cminus_max'),seams=result['topology_closure'],failed_gates={k:v for k,v in report.items() if k.endswith('passes') and not v})),flush=True)
    return result


if __name__ == '__main__':
    cases=[]
    for weight in (.05,1.):
        for budget in (100,400,1000):
            cases.append((f"t02_n10_w{weight:g}_budget{budget}",replace(BASE,physics_weight=weight,max_nfev=budget)))
    blocks=tuple(v for v in CHARACTERISTIC_RAO_RESIDUAL_BLOCKS if v!='regularization')
    for n in (10,16):
        cases.append((f"t02_n{n}_w1_budget400_no_regularization",replace(BASE,n_control=n,physics_weight=1.,max_nfev=400,residual_blocks=blocks)))
    for label,config in cases:
        if (OUT/f"{label}.json").exists():
            continue
        run_case(label,config)
