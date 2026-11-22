"""E12b (POST-HOC; E12 original status FAIL is unchanged): Taylor test of the JAX
throat-kernel theta_B derivative at the midpoints of event-free intervals, where an
event is any recorded change of row count, raw/special pattern or unit-process
branch flag on the final BD row.  Also lists the branch guards on the kernel path
(planner answer A9).  Approved as opus_a answers_round1 A10.

.venv-jax/bin/python tests/experiments/nozzle/moc_ext_e12_taylor_posthoc.py
"""
from __future__ import annotations

from lrekit.paths import repository_root as _repository_root

import argparse
import ast
import json
import math
import os
import sys
import time

import numpy as np

ROOT = str(_repository_root())
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests", "experiments", "nozzle"))

import lrekit.numerics.jax  # noqa: E402,F401
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from lrekit.numerics.jax import assembly  # noqa: E402
from lrekit.numerics.jax.moc_kernel import march_kernel  # noqa: E402
from lrekit.numerics.jax.theta_b_solve import _n_rows_bound, start_line_from_kernel  # noqa: E402
from lrekit.components.nozzle.solvers.nasa import build_kernel  # noqa: E402

import moc_ext_e12_jax_kernel_taylor as e12  # noqa: E402
import moc_ext_e12_branch_switch_diagnostic as diag  # noqa: E402

PLAN = os.path.join(ROOT, "builds", "moc_extensions_plan_2026-11-21", "opus_b")
E12_DIR = os.path.join(PLAN, "e12")
OUT_DEFAULT = os.path.join(PLAN, "e12b")
FLAG_NAMES = ["T2_zform_vs_rform", "T1_zform_vs_rform", "r3_slope_select", "TH3_clamp_zero", "theta2_bad_guard",
              "r3_clamp_zero"]
KERNEL_PATH = {
    "raosim/jax/moc_kernel.py": ["_mm", "_coef_A", "_guarded_inv", "_coef_B", "_coef_b", "_coef_R", "_coef_Rs",
                                 "_tan_avg", "_mu_of", "interior_point_nasa", "axis_point_nasa",
                                 "arc_wall_point_raw_nasa", "special_wall_point_nasa", "_sweep_row_padded",
                                 "march_kernel"],
    "raosim/jax/primitives.py": ["prandtl_meyer", "mach_angle", "isentropic_temperature_ratio",
                                 "isentropic_density_ratio"],
    "raosim/jax/theta_b_solve.py": ["start_line_from_kernel"],
    "raosim/jax/assembly.py": ["_polyline_mass_flux"],
}
GUARD_CALLS = {"where": "select", "maximum": "floor/clamp", "minimum": "ceiling/clamp", "clip": "clip",
               "select": "select", "cond": "branch"}


def guard_inventory():
    out = []
    for rel, funcs in KERNEL_PATH.items():
        src = open(os.path.join(ROOT, rel)).read()
        lines = src.splitlines()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in funcs:
                seen = set()
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) \
                            and sub.func.attr in GUARD_CALLS and sub.lineno not in seen:
                        seen.add(sub.lineno)
                        text = lines[sub.lineno - 1].strip()
                        entry = dict(location=f"{rel}:{sub.lineno}", function=node.name,
                                     kind=GUARD_CALLS[sub.func.attr], source=text)
                        if "TH3 = jnp.maximum(TH3, 0.0)" in text:
                            entry["note"] = ("near-axis TH3 >= 0 clamp; switches inside row-count windows in E12 "
                                             "(observation (c) of MOC-27)")
                        out.append(entry)
    out.sort(key=lambda e: (e["location"].split(":")[0], int(e["location"].split(":")[1])))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunk", type=int, default=43)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()
    res12 = json.load(open(os.path.join(E12_DIR, "result.json")))
    npz_path = os.path.join(E12_DIR, "e12_sweep.npz")
    sw = np.load(npz_path)
    theta_deg = sw["theta_B_deg"]
    theta_rad = np.radians(theta_deg)
    n = theta_deg.size
    k_hi = build_kernel(e12.RT, e12.RD, math.radians(e12.TB_HI_DEG), e12.GAMMA, e12.N_KERNEL,
                        starting_line_method="kliegel_levine", max_rrcs=500, dtheta_limit=e12.DTL, Ru=e12.RU)
    sl = start_line_from_kernel(k_hi)
    n_rows = _n_rows_bound(math.radians(e12.TB_HI_DEG), e12.DTL) + e12.ROW_MARGIN
    assert e12.sha256_arrays(*sl) == res12["inputs"]["start_line_sha256"]

    def outputs(tb):
        m = march_kernel(sl, tb, e12.RT, e12.RD, e12.GAMMA, n_rows=n_rows, n_corr=40, dtheta_limit=e12.DTL)
        bd = m.bd
        j1 = assembly._polyline_mass_flux(bd.x, bd.r, bd.M, bd.theta, e12.GAMMA, assembly._CE_M_FLOOR)
        idx = jnp.argmax(m.wall_theta[1:] >= tb - 1e-9) + 1
        prev = jax.tree.map(lambda a: a[idx - 1], m.rows)
        aux = dict(reached=m.reached_wall, axis_idx=m.bd_axis_idx, row_axis_idx=m.row_axis_idx,
                   wall_theta=m.wall_theta, bd=bd, prev=prev)
        return jnp.stack([j1, bd.M[0], bd.x[m.bd_axis_idx]]), aux

    def with_tangent(tb):
        return jax.jvp(outputs, (tb,), (jnp.ones_like(tb),), has_aux=True)

    def state(aux, k=None):
        g = (lambda a: np.asarray(a)) if k is None else (lambda a: np.asarray(a)[k])
        bd = {nm: g(getattr(aux["bd"], nm)) for nm in ("x", "r", "theta", "M")}
        prev = {nm: g(getattr(aux["prev"], nm)) for nm in ("x", "r", "theta", "M")}
        ax = int(g(aux["axis_idx"]))
        return bd, prev, ax, g(aux["row_axis_idx"]), g(aux["wall_theta"]), bool(g(aux["reached"]))

    def signature(tb, aux, k=None):
        bd, prev, ax, rai, wt, reached = state(aux, k)
        pat = e12.pattern_of(tb, rai, wt)
        fl = diag.branch_flags(bd, prev, ax)
        return (pat.get("signature"), fl.tobytes(), fl.shape), reached

    sweep = jax.jit(jax.vmap(outputs))
    n_chunks = int(math.ceil(n / args.chunk))
    tb_pad = np.concatenate([theta_rad, np.full(n_chunks * args.chunk - n, theta_rad[-1])])
    t0 = time.time()
    sweep_c = sweep.lower(jnp.asarray(tb_pad[:args.chunk])).compile()
    t_compile_sweep = time.time() - t0
    t0 = time.time()
    sigs = []
    for c in range(n_chunks):
        _, aux = sweep_c(jnp.asarray(tb_pad[c * args.chunk:(c + 1) * args.chunk]))
        for k in range(args.chunk):
            if len(sigs) < n:
                sigs.append(signature(theta_rad[len(sigs)], aux, k))
    t_eval_sweep = time.time() - t0
    pattern_change = sw["pattern_change"]
    any_change = np.array([sigs[i][0] != sigs[i + 1][0] for i in range(n - 1)])
    assert np.all(any_change[pattern_change])
    events = np.nonzero(any_change)[0]
    intervals = []
    for a_, b_ in zip(events[:-1], events[1:]):
        lo, hi = theta_deg[a_ + 1], theta_deg[b_]
        if hi > lo:
            intervals.append(dict(lo_deg=float(lo), hi_deg=float(hi), mid_deg=float(0.5 * (lo + hi)),
                                  h_max_deg=float(0.5 * (hi - lo)),
                                  after_event=("row_count_or_pattern" if pattern_change[a_] else "branch_flag")))
    windows = res12["windows"]
    chosen = []
    for tgt in e12.TAYLOR_TARGETS_DEG:
        w = next(w for w in windows if w["theta_lo_deg"] <= tgt <= w["theta_hi_deg"])
        cand = [iv for iv in intervals if iv["lo_deg"] >= w["theta_lo_deg"] - 1e-12 and iv["hi_deg"] <= w["theta_hi_deg"] + 1e-12]
        chosen.append(dict(max(cand, key=lambda iv: iv["hi_deg"] - iv["lo_deg"]), target_deg=tgt,
                           window=[w["theta_lo_deg"], w["theta_hi_deg"]]))

    single = jax.jit(outputs)
    single_t = jax.jit(with_tangent)
    t0 = time.time()
    single_c = single.lower(jnp.float64(theta_rad[0])).compile()
    single_t_c = single_t.lower(jnp.float64(theta_rad[0])).compile()
    t_compile_taylor = time.time() - t0
    t0 = time.time()
    taylor = []
    for iv in chosen:
        t_deg = iv["mid_deg"]
        t = math.radians(t_deg)
        J0, dJ0, aux0 = single_t_c(jnp.float64(t))
        J0, dJ0 = np.asarray(J0), np.asarray(dJ0)
        sig0, reached0 = signature(t, aux0)
        hs = [h for h in e12.H_DEG if h <= iv["h_max_deg"] + 1e-15]
        rows = []
        for hd in hs:
            h = math.radians(hd)
            row = dict(h_deg=hd)
            for sgn, key in ((1.0, "plus"), (-1.0, "minus")):
                Jh, auxh = single_c(jnp.float64(t + sgn * h))
                sigh, rh = signature(t + sgn * h, auxh)
                row[f"E_{key}"] = [float(e) for e in np.abs(np.asarray(Jh) - J0 - sgn * h * dJ0)]
                row[f"same_state_{key}"] = bool(sigh == sig0 and rh)
            rows.append(row)
        per = {}
        for q, nm in enumerate(e12.J_NAMES):
            fits = {}
            for key in ("plus", "minus"):
                E = np.array([r[f"E_{key}"][q] for r in rows])
                h = np.array(hs)
                ok = np.array([r[f"same_state_{key}"] for r in rows])
                small = h <= 1e-6
                noise = float(np.max(E[small])) if np.any(small) else 0.0
                usable = ok & (E >= e12.NOISE_FLOOR_FACTOR * noise) & (E > 0)
                idx = np.nonzero(usable)[0]
                run = []
                if idx.size:
                    run = [idx[0]]
                    for k in idx[1:]:
                        if k == run[-1] + 1:
                            run.append(k)
                        else:
                            break
                if len(run) >= 2:
                    lh, lE = np.log10(h[run]), np.log10(E[run])
                    slope, dec = float(np.polyfit(lh, lE, 1)[0]), float(lh.max() - lh.min())
                else:
                    slope, dec = None, 0.0
                fits[key] = dict(E=E.tolist(), noise_floor=noise, fit_h_deg=[float(h[k]) for k in run],
                                 fitted_slope=slope, fit_decades=dec,
                                 passes=bool(slope is not None and slope >= 1.8 and dec >= 2.0 - 1e-12))
            per[nm] = dict(J=float(J0[q]), dJ_dthetaB_per_rad=float(dJ0[q]), forward=fits["plus"], backward=fits["minus"],
                           passes=fits["plus"]["passes"])
        taylor.append(dict(theta_B_deg=t_deg, interval=iv, h_deg=hs, rows=rows, outputs=per, reached=reached0))
    t_eval_taylor = time.time() - t0

    J = sw["J"]
    g = e12.GAMMA
    mdot_ideal = math.pi * (2.0 / (g + 1.0)) ** (1.0 / (g - 1.0)) * math.sqrt(g * 2.0 / (g + 1.0))
    j1_ratio = float(np.mean(J[sw["reached_wall"], 0]) / mdot_ideal)
    guards = guard_inventory()

    plt = e12.style()
    fig, ax = plt.subplots(1, 1, figsize=(3.2, 2.8))
    colors = [e12.INK, e12.TEAL, e12.ACCENT]
    markers = ["o", "s", "^"]
    for p_, tp in enumerate(taylor):
        h = np.array(tp["h_deg"])
        for q, nm in enumerate(e12.J_NAMES):
            E = np.array(tp["outputs"][nm]["forward"]["E"])
            m = E > 0
            ax.loglog(h[m], E[m], "-", marker=markers[p_], ms=2.8, lw=0.7, color=colors[q])
    href = np.array([1e-4, 1e-2])
    anchor = taylor[0]["outputs"]["J3_x_D_axis"]["forward"]["E"][0]
    ax.loglog(href, anchor * (href / 1e-2) ** 2 * 3, "--", color=e12.MUTED, lw=0.8)
    ax.text(3e-4, anchor * (3e-4 / 1e-2) ** 2 * 12, "slope 2", color=e12.MUTED, fontsize=7.5)
    for q, lab in enumerate(["$J_1$", "$J_2$", "$J_3$"]):
        ax.plot([], [], "-", color=colors[q], lw=0.8, label=lab)
    for p_, tp in enumerate(taylor):
        ax.plot([], [], ls="none", marker=markers[p_], color=e12.MUTED, ms=3,
                label=rf"$\theta_B$={tp['theta_B_deg']:.4f}$^\circ$")
    ax.set_xlabel(r"step $h$ (deg)")
    ax.set_ylabel(r"$E(h)=|J(t+h)-J(t)-hJ'(t)|$")
    ax.legend(loc="lower right", frameon=False, fontsize=6.0, ncol=2, handlelength=1.2, columnspacing=0.6)
    fig.tight_layout()
    base = os.path.join(args.out, "e12b_taylor_posthoc")
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=200)
    plt.close(fig)

    all_pass = all(tp["outputs"][nm]["passes"] for tp in taylor for nm in e12.J_NAMES)
    script = os.path.abspath(__file__)
    result = dict(
        id="E12b", status=("PASS" if all_pass else "FAIL") + " (POST-HOC)", labels=["POST-HOC", "E12 original status FAIL"],
        evidence_class="verification (post hoc; cannot change E12's status)",
        command=" ".join([".venv-jax/bin/python", os.path.relpath(script, ROOT)] + list(argv if argv is not None else sys.argv[1:])),
        git_head=e12.git_head(), script=os.path.relpath(script, ROOT), script_sha256=e12.sha256_file(script),
        raosim_sources_sha256=e12.raosim_source_hashes(),
        inputs={os.path.relpath(npz_path, ROOT): e12.sha256_file(npz_path),
                os.path.relpath(os.path.join(E12_DIR, "result.json"), ROOT): e12.sha256_file(os.path.join(E12_DIR, "result.json")),
                "tests/experiments/nozzle/moc_ext_e12_jax_kernel_taylor.py": e12.sha256_file(e12.__file__),
                "tests/experiments/nozzle/moc_ext_e12_branch_switch_diagnostic.py": e12.sha256_file(diag.__file__)},
        configuration=dict(same_as="E12 (start line, n_rows, n_corr, dtheta_limit, outputs J1-J3, h grid)",
                           event_definition=("any change between consecutive sweep points of the active row count, "
                                             "raw/special flag string or any of the branch flags " + ", ".join(FLAG_NAMES)
                                             + " on the final BD row"),
                           target_rule=("for each E12 target (27.25, 27.75, 28.25 deg): the longest event-free interval "
                                        "inside the row-count window containing the target; t = its midpoint; h <= half "
                                        "its length so that t +/- h stays inside"),
                           fit_rule="as E12 (forward E(h); 100 x noise floor at h <= 1e-6 deg; slope >= 1.8 over >= 2 decades)",
                           approval="opus_a answers_round1 A10"),
        timings=dict(sweep_jit_compile_s=t_compile_sweep, sweep_eval_s=t_eval_sweep,
                     taylor_jit_compile_s=t_compile_taylor, taylor_eval_s=t_eval_taylor),
        wall_time_s=time.time() - t_start,
        metrics=dict(n_events=int(events.size), n_row_count_or_pattern_events=int(pattern_change.sum()),
                     n_branch_flag_only_events=int((any_change & ~pattern_change).sum()),
                     n_event_free_intervals=len(intervals),
                     taylor_summary=[dict(theta_B_deg=tp["theta_B_deg"], h_max_deg=tp["interval"]["h_max_deg"],
                                          **{nm: dict(slope=tp["outputs"][nm]["forward"]["fitted_slope"],
                                                      decades=tp["outputs"][nm]["forward"]["fit_decades"],
                                                      noise_floor=tp["outputs"][nm]["forward"]["noise_floor"],
                                                      passes=tp["outputs"][nm]["passes"],
                                                      backward_slope=tp["outputs"][nm]["backward"]["fitted_slope"],
                                                      backward_decades=tp["outputs"][nm]["backward"]["fit_decades"])
                                             for nm in e12.J_NAMES}) for tp in taylor],
                     J1_noise_floor=[tp["outputs"]["J1_mdot_BD"]["forward"]["noise_floor"] for tp in taylor],
                     all_pass=all_pass),
        event_free_intervals=intervals, taylor=taylor,
        moc29_evidence=dict(J1_mean_over_1D_ideal_choked_mass_flow=j1_ratio, ideal_nondimensional=mdot_ideal,
                            note=("E12 sweep, n_kernel 24, dtheta 0.25 deg, KL start line: BD mass flow exceeds the "
                                  "1-D ideal; recorded as evidence for MOC-29 (production start line C_D > 1) per A10")),
        branch_guard_inventory=dict(
            answer="A9 (planner policy decision: N2 tracks unit-process branch state)",
            scope="functions on the march_kernel path, the start-line converter and the J1 mass-flux routine",
            n_guards=len(guards), guards=guards,
            th3_clamp_observation=("MOC-27 observation (c): the near-axis TH3 = max(TH3, 0) clamp in "
                                   "interior_point_nasa switches inside row-count windows (E12 diagnostic: 12 switches "
                                   "in 27.0-28.5 deg, about 0.02 deg after each window start; post-hoc band, this "
                                   "configuration only, not a policy)")),
        figures={"e12b_taylor_posthoc": {"pdf": os.path.relpath(base + ".pdf", ROOT), "pdf_sha256": e12.sha256_file(base + ".pdf"),
                                         "png": os.path.relpath(base + ".png", ROOT), "png_sha256": e12.sha256_file(base + ".png")}},
        caption_facts=dict(
            configuration=("POST-HOC Taylor remainders of the JAX throat-kernel outputs J1-J3 at the midpoints of the "
                           "longest event-free theta_B intervals in three row-count windows (Nozzle-B, n 24, dtheta "
                           "0.25 deg); steps limited to stay inside the interval."),
            evidence_class="verification (post hoc)",
            not_established="E12's pre-registered result (unchanged FAIL); derivatives of the full net, start line or BVP."),
        notes=["E12 remains FAIL; this record cannot change it."],
    )
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(result, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(dict(status=result["status"], metrics=result["metrics"], moc29=result["moc29_evidence"],
                          n_guards=len(guards)), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
