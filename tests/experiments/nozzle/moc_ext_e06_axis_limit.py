"""E06: exact-axis source-limit probe of lrekit.components.nozzle.solvers.moc.solve_axis_point (MOC-14).

Unit level: p_above on the exact source flow of E05 (tests/experiments/nozzle/moc_ext_e05_source_flow_oracle.py)
at fixed x and r_above -> 0; compare the axis point with the exact axis foot of the exact C-.
Instrumentation: during the E05 bounded-net runs, wrap solve_axis_point and
rao_residuals._source_axisym (and the NASA axial process the bounded net actually calls)
and count small-r branch hits. Opt-in diagnostic; changes nothing in raosim/.

    .venv-jax/bin/python tests/experiments/nozzle/moc_ext_e06_axis_limit.py
"""
from __future__ import annotations

from lrekit.paths import repository_root as _repository_root

import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = _repository_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests/experiments/nozzle"))
OUT = ROOT / "builds/moc_extensions_plan_2026-11-21/opus_b/e06"

import numpy as np  # noqa: E402

import moc_ext_e05_source_flow_oracle as e05  # noqa: E402
import lrekit.components.nozzle.solvers.moc as moc  # noqa: E402
import lrekit.components.nozzle.solvers.bounded as mb  # noqa: E402
import lrekit.components.nozzle.solvers.nasa as nm  # noqa: E402
import lrekit.components.nozzle.solvers.residuals as rr  # noqa: E402

INK, TEAL, MUTED, ACCENT = "#19333C", "#176D76", "#51616B", "#C26A2E"
R_LIST = (0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
GAMMAS = (1.23, 1.4)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unit_probe(g: float):
    x_fixed = math.sqrt(e05.area_ratio(2.0, g))
    rows = []
    for r_above in R_LIST:
        M_a, th_a = e05.exact_state(x_fixed, r_above, g)
        p = nm._char_point_from_values(x_fixed, r_above, th_a, M_a, g)
        x_ex = e05.exact_cminus_axis_foot(x_fixed, r_above, g, dx=r_above / 4000.0)
        M_ex = e05.mach_of_R(x_ex, g)
        q = moc.solve_axis_point(p, g)
        n = nm._calc_axial_mesh_point([p], g)
        rows.append({"r_above": r_above, "x_above": x_fixed, "M_above": M_a, "theta_above_rad": th_a,
                     "x_exact": x_ex, "M_exact": M_ex,
                     "solve_axis_point": {"x3": q.x, "M3": q.M, "dx": q.x - x_ex, "dM": q.M - M_ex},
                     "nasa_calc_axial_mesh_point": {"x3": n.x, "M3": n.M, "dx": n.x - x_ex, "dM": n.M - M_ex}})
    def order(key, q):
        v = [abs(r[key][q]) for r in rows]
        rs = [r["r_above"] for r in rows]
        return [math.log(a / b) / math.log(ra / rb) if a > 0 and b > 0 else None
                for a, b, ra, rb in zip(v[:-1], v[1:], rs[:-1], rs[1:])]
    return {"gamma": g, "x_fixed": x_fixed, "rows": rows,
            "orders": {k: {q: order(k, q) for q in ("dM", "dx")}
                       for k in ("solve_axis_point", "nasa_calc_axial_mesh_point")}}


class Counter:
    def __init__(self):
        self.calls, self.min_r, self.small_r_hits, self.zero_branch_hits = 0, math.inf, 0, 0


def instrumented_e05_runs(levels=(13, 25, 49, 97, 193)):
    orig = {"solve_axis_point": moc.solve_axis_point, "_source_axisym": rr._source_axisym,
            "_calc_axial_mesh_point": nm._calc_axial_mesh_point, "_nasa_calc_b": nm._nasa_calc_b}
    c = {k: Counter() for k in orig}

    def w_solve_axis_point(p_above, gamma, *a, **k):
        cc = c["solve_axis_point"]
        cc.calls += 1
        cc.min_r = min(cc.min_r, float(p_above.r))
        if not p_above.r > 1e-10:
            cc.small_r_hits += 1
        if 0.5 * p_above.r <= 1e-12:
            cc.zero_branch_hits += 1
        return orig["solve_axis_point"](p_above, gamma, *a, **k)

    def w_source(theta, mu, r):
        cc = c["_source_axisym"]
        cc.calls += 1
        cc.min_r = min(cc.min_r, float(r))
        if r <= 1e-12:
            cc.zero_branch_hits += 1
        return orig["_source_axisym"](theta, mu, r)

    def w_axial(new_wall_first, gamma, *a, **k):
        cc = c["_calc_axial_mesh_point"]
        cc.calls += 1
        if new_wall_first:
            r2 = float(new_wall_first[-1].r)
            cc.min_r = min(cc.min_r, r2)
            if r2 <= 1e-12:
                cc.zero_branch_hits += 1
        return orig["_calc_axial_mesh_point"](new_wall_first, gamma, *a, **k)

    def w_b(mach, theta, r):
        cc = c["_nasa_calc_b"]
        cc.calls += 1
        cc.min_r = min(cc.min_r, float(r))
        if r == 0.0:
            cc.zero_branch_hits += 1
        return orig["_nasa_calc_b"](mach, theta, r)

    wrappers = {"solve_axis_point": w_solve_axis_point, "_source_axisym": w_source,
                "_calc_axial_mesh_point": w_axial, "_nasa_calc_b": w_b}
    patched = []
    for name, mod in list(sys.modules.items()):
        if not name.startswith("lrekit"):
            continue
        for attr, fn in orig.items():
            if getattr(mod, attr, None) is fn:
                setattr(mod, attr, wrappers[attr])
                patched.append(f"{name}.{attr}")
    try:
        runs = []
        for a, g in ((15.0, 1.23), (15.0, 1.4), (25.0, 1.23), (25.0, 1.4)):
            for n in levels:
                _, _, _, net, wt, err = e05.run_level(a, g, n)
                runs.append({"alpha_deg": a, "gamma": g, "n": n, "complete": bool(net and net.complete)})
    finally:
        for name, mod in list(sys.modules.items()):
            if not name.startswith("lrekit"):
                continue
            for attr, fn in orig.items():
                if getattr(mod, attr, None) is wrappers[attr]:
                    setattr(mod, attr, fn)
    counts = {k: {"calls": v.calls, "min_r": (None if v.min_r == math.inf else v.min_r),
                  "zero_branch_hits": v.zero_branch_hits,
                  **({"r_above_le_1e-10_hits": v.small_r_hits} if k == "solve_axis_point" else {})}
              for k, v in c.items()}
    return {"patched_bindings": sorted(patched), "runs": runs, "counts": counts}


def figure(probes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "axes.labelsize": 8.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
                         "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "pdf.fonttype": 42})
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.7), constrained_layout=True)
    cols = {1.23: TEAL, 1.4: INK}
    for pr in probes:
        r = np.array([x["r_above"] for x in pr["rows"]])
        for key, ls, mk, lab in (("solve_axis_point", "-", "o", "solve_axis_point"),
                                 ("nasa_calc_axial_mesh_point", "--", "s", "NASA axial point")):
            dM = np.array([abs(x[key]["dM"]) for x in pr["rows"]])
            dx = np.array([abs(x[key]["dx"]) for x in pr["rows"]])
            kw = dict(ls=ls, marker=mk, ms=3.5, lw=1.0, color=cols[pr["gamma"]],
                      mfc="white" if ls == "--" else cols[pr["gamma"]])
            axs[0].loglog(r, dM, label=f"{lab}, $\\gamma$ = {pr['gamma']}", **kw)
            axs[1].loglog(r, dx, **kw)
    for ax, q in zip(axs, ("dM", "dx")):
        ref = probes[0]["rows"]
        r = np.array([x["r_above"] for x in ref])
        y0 = 0.3 * abs(ref[-1]["solve_axis_point"][q])
        ax.loglog(r, y0 * (r / r[-1]), color=MUTED, ls=":", lw=0.8, label="slope 1")
        ax.loglog(r, y0 * (r / r[-1]) ** 2, color=ACCENT, ls=":", lw=0.8, label="slope 2")
        ax.set_xlabel(r"$r_{\rm above}/R^*$ (-)")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].set_ylabel(r"$|M_3 - M_{\rm exact}|$ (-)")
    axs[1].set_ylabel(r"$|x_3 - x_{\rm exact}|/R^*$ (-)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    paths = []
    for ext in ("pdf", "png"):
        p = OUT / f"e06_axis_limit.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(str(p.relative_to(ROOT)))
    plt.close(fig)
    return paths


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    probes = [unit_probe(g) for g in GAMMAS]
    inst = instrumented_e05_runs()
    figs = figure(probes)
    sap_orders = [o for pr in probes for q in ("dM", "dx") for o in pr["orders"]["solve_axis_point"][q]]
    sap_mono = all(abs(a["solve_axis_point"][q]) > abs(b["solve_axis_point"][q])
                   for pr in probes for q in ("dM", "dx") for a, b in zip(pr["rows"][:-1], pr["rows"][1:]))
    min_order = min(o for o in sap_orders if o is not None)
    zero_hits_sap = inst["counts"]["solve_axis_point"]["zero_branch_hits"]
    zero_hits_src = inst["counts"]["_source_axisym"]["zero_branch_hits"]
    status = "PASS" if (sap_mono and min_order >= 1.0 and zero_hits_sap == 0 and zero_hits_src == 0) else "FAIL"
    findings = []
    if inst["counts"]["solve_axis_point"]["calls"] == 0:
        findings.append("lrekit.components.nozzle.solvers.bounded.march_bounded_net never calls lrekit.components.nozzle.solvers.moc.solve_axis_point or "
                        "rao_residuals._source_axisym (0 calls in all E05 runs); its axis points come from "
                        "nasa_moc._calc_axial_mesh_point, which evaluates the source term only at the off-axis "
                        "parent (b2 of p2). The 'zero hits in E05' half of the hypothesis therefore holds "
                        "vacuously for the spec'd functions.")
    findings.append("solve_axis_point does not call _source_axisym: it computes S inline with "
                    "r_avg = max(0.5 r_above, 1e-12) and skips the source term when r_above <= 1e-10.")
    findings.append("Static callers of solve_axis_point: raosim/moc.py:547,549 (legacy front march), "
                    "raosim/nasa_moc.py:2766 (_calc_remaining_mesh_row), raosim/rao_variational.py:2379,2381,"
                    "2697,2699. None is on the march_bounded_net path exercised by E05.")
    findings.append("Smallest off-axis parent radius handed to the bounded net's axial process in the E05 runs "
                    f"was {inst['counts']['_calc_axial_mesh_point']['min_r']:.3e} R* (the exit-lip closure "
                    "row of cone 15 deg, gamma 1.23, n = 193; see E05 findings).")
    result = {
        "id": "E06", "status": status,
        "evidence_class": "verification",
        "configuration": {
            "exact_field": "E05 radial source flow (R* = 1); A/A* = (R/R*)^2 derivation, see E05 (LR6)",
            "gammas": list(GAMMAS), "r_above_over_Rstar": list(R_LIST),
            "x_fixed": "x = R(M = 2) for each gamma, i.e. sqrt(A/A*(2, gamma))",
            "p_above": "nasa_moc._char_point_from_values(x, r, theta_exact, M_exact, gamma) (compat_plus = theta + nu)",
            "exact_axis_foot": "RK4 in x along the exact C- (dr/dx = tan(theta - mu)), dx = r_above/4000",
            "functions": ["lrekit.components.nozzle.solvers.moc.solve_axis_point (spec)",
                          "lrekit.components.nozzle.solvers.nasa._calc_axial_mesh_point (supplementary: the axis process the bounded "
                          "net actually calls)"],
            "instrumentation": "wrappers bound in every raosim module that holds the original function object, "
                               "during E05 runs (cones 15/25 deg, gamma 1.23/1.4, n = 13..193)",
        },
        "command": ".venv-jax/bin/python tests/experiments/nozzle/moc_ext_e06_axis_limit.py",
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                   text=True).stdout.strip(),
        "hashes": {"script": {str(Path(__file__).resolve().relative_to(ROOT)): sha256(Path(__file__)),
                              "tests/experiments/nozzle/moc_ext_e05_source_flow_oracle.py":
                                  sha256(ROOT / "tests/experiments/nozzle/moc_ext_e05_source_flow_oracle.py")},
                   "raosim_imported": e05.raosim_hashes(), "inputs": {}},
        "metrics": {"unit_probe": probes, "instrumentation": inst,
                    "solve_axis_point_min_observed_order": min_order,
                    "solve_axis_point_errors_monotone": sap_mono},
        "hypotheses": {
            "H_E06_zero_branch_never_hit": ("supported (vacuously for the bounded net: the spec'd functions are "
                                            "never called)" if zero_hits_sap == 0 and zero_hits_src == 0
                                            else "not supported"),
            "H_E06_error_order_ge_1": "supported" if (sap_mono and min_order >= 1.0) else "not supported",
        },
        "findings": findings,
        "figures": [{"paths": figs, "caption_facts": {
            "configuration": "Axis unit process fed with exact source-flow data at fixed x = R(M = 2) and "
                             "r_above from 0.3 to 0.001 R*, gamma 1.23 and 1.4; error of the computed axis "
                             "point against the exact axis foot of the exact C-; solid = moc.solve_axis_point, "
                             "dashed = nasa_moc._calc_axial_mesh_point.",
            "evidence_class": "verification",
            "not_established": "Behaviour of the JAX unit processes or the BDE backward point at the axis; "
                               "error accumulation over a full net (see E05)."}}],
        "wall_time_s": time.time() - t0,
        "notes": [],
    }
    for _f in result["figures"]:
        _f.update({"script": str(Path(__file__).resolve().relative_to(ROOT)), "command": result["command"],
                   "git_head": result["git_head"], "status": result["status"],
                   "wall_time_s": result["wall_time_s"], "hashes": "see result.hashes (script, raosim, inputs)",
                   "file_sha256": {q: sha256(ROOT / q) for q in _f["paths"]}})
    (OUT / "result.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"status": status, "orders": [pr["orders"] for pr in probes],
                      "counts": inst["counts"], "patched": inst["patched_bindings"]}, indent=1))


if __name__ == "__main__":
    main()
