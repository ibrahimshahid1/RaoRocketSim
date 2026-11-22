"""E08b (report-only): E08 separation-criteria sweep re-run on the E07b forward-net wall states.

Opt-in experiment script (moc_extensions_plan_2026-11-21). Runs only because E07b ended PASS-DIAGNOSTIC.
Run: .venv-jax/bin/python -m tests.experiments.nozzle.moc_ext_e08b_separation_forward
"""

from __future__ import annotations

import json
import math

import numpy as np

from scripts import moc_ext_e00_common as C


def forward_states(path):
    from lrekit.core.gas_dynamics import isentropic_pressure_ratio
    d = np.load(path)
    x, M = np.asarray(d["x"], float), np.asarray(d["M"], float)
    order = np.argsort(x, kind="stable")
    x, M = x[order], M[order]
    keep = np.concatenate([[True], np.diff(x) > 0.0])
    x, M = x[keep], M[keep]
    return x, M, np.array([isentropic_pressure_ratio(m, C.GAMMA) for m in M])


def main():
    from lrekit.core.gas_dynamics import isentropic_pressure_ratio, mach_from_area_ratio
    from tests.experiments.nozzle.moc_ext_e08_separation_sweep import (CRITERIA, LABEL, P0_OVER_PA, design_ratio, incipient,
                                                     spot_checks, sweep)
    run = C.Run("E08b", __file__)
    e07b = C.OUT_ROOT / "e07b" / "result.json"
    e08 = C.OUT_ROOT / "e08" / "result.json"
    run.add_input(e07b)
    run.add_input(e08)
    status_e07b = json.load(open(e07b))["status"]
    if status_e07b != "PASS-DIAGNOSTIC":
        run.finish({"status": "SKIPPED-CONDITION", "reason": f"E07b status {status_e07b} (needs PASS-DIAGNOSTIC)"})
        print("E08b skipped")
        return
    e08_inc = json.load(open(e08))["metrics"]["incipient_p0_over_pa"]
    spots = spot_checks()
    eps = 4.973
    Me_q = mach_from_area_ratio(eps, C.GAMMA, supersonic=True)
    pe_q = isentropic_pressure_ratio(Me_q, C.GAMMA)
    lv = {}
    for lvl in ("refined", "baseline"):
        path = C.OUT_ROOT / "e07b" / f"forward_wall_{lvl}.npz"
        run.add_input(path)
        x, M, p = forward_states(path)
        crit = {}
        for n in CRITERIA:
            xs, flags = sweep(x, M, p, n)
            iw = incipient(n, M[-1], p[-1])
            bvp = e08_inc[lvl][n]["wall"]
            crit[n] = {"incipient_wall": iw, "design_wall_reserve_1.2": design_ratio(n, M[-1], p[-1]),
                       "incipient_bvp_wall_E08": bvp,
                       "change_vs_E08_rel": (iw - bvp) / bvp if (bvp and math.isfinite(iw)) else None,
                       "grid_points_outside_validity": int(sum(f == "outside_validity" for f in flags)),
                       "x_sep": xs}
        lv[lvl] = {"n_stations": int(len(x)), "M_exit": float(M[-1]), "p_exit": float(p[-1]),
                   "p_monotone_decreasing": bool(np.all(np.diff(p) <= 0.0)), "criteria": crit}
    ref = lv["refined"]["criteria"]
    main_set = ("summerfield", "schilling_0582", "kalt_badal", "schmucker", "tp1207")
    vals = np.array([ref[n]["incipient_wall"] for n in main_set])
    spread = float(np.nanmax(vals) / np.nanmin(vals))
    sch_q = incipient("schmucker", Me_q, pe_q)
    figs = make_figure(run, lv)
    result = {
        "status": "PASS" if all(s["passes"] for s in spots) else "FAIL",
        "labels": ["report-only", "input: E07b forward-net wall states (exit-plane-limited harness, DIAGNOSTIC)"],
        "evidence_class": "neither (screening-criteria comparison on an inviscid wall)",
        "configuration": {"wall_states": "E07b forward-net wall points (rows[j][0]) of the 48/64 replay (32/48 cross-check)",
                          "method": "identical to E08 (functions imported from tests/experiments/nozzle/moc_ext_e08_separation_sweep.py)"},
        "metrics": {
            "spot_checks_max_rel": max(s["rel"] for s in spots),
            "levels": {l: {k: v for k, v in d.items() if k != "criteria"} for l, d in lv.items()},
            "incipient_p0_over_pa": {l: {n: {k: c[k] for k in c if k != "x_sep"} for n, c in d["criteria"].items()}
                                     for l, d in lv.items()},
            "spread_incipient_wall_max_over_min": spread,
            "schmucker_wall_vs_q1d_incipient_rel": (ref["schmucker"]["incipient_wall"] - sch_q) / sch_q,
            "max_abs_change_vs_E08_rel_refined": max(abs(c["change_vs_E08_rel"]) for c in ref.values()
                                                    if c["change_vs_E08_rel"] is not None),
        },
        "hypotheses": {
            "H_criteria_spread_exceeds_reserve": "supported" if spread > 1.2 else "not supported",
            "H_schmucker_wall_differs_from_mdo_q1d": "supported"
            if abs((ref["schmucker"]["incipient_wall"] - sch_q) / sch_q) > 0.01 else "not supported",
        },
        "figures": figs,
        "caption_facts": {
            "configuration": ("E08 criteria applied to the E07b exit-plane-limited forward-net wall states of the accepted "
                              "t19 Nozzle-B (48/64; gamma 1.23, vacuum design used as a wall-state carrier), compared "
                              "with the E08 values on the BVP wall states."),
            "evidence_class": "neither (screening-criteria comparison on an inviscid wall)",
            "not_established": "Which criterion is physically right; RSS, hysteresis or side loads; any validation.",
        },
        "not_established": "Which criterion is physically right; RSS, hysteresis or side loads; any validation.",
    }
    run.finish(result, extra_scripts=[C.REPO / "tests/experiments/nozzle/moc_ext_e08_separation_sweep.py"])
    print("E08b", result["status"], "spread", spread, "max change", result["metrics"]["max_abs_change_vs_E08_rel_refined"])
    for n in CRITERIA:
        print(f"  {n:16s} {ref[n]['incipient_wall']:.5f} vs E08 {ref[n]['incipient_bvp_wall_E08']:.5f} "
              f"rel {ref[n]['change_vs_E08_rel']:.3e}")


def make_figure(run, lv):
    plt = C.setup_style()
    from tests.experiments.nozzle.moc_ext_e08_separation_sweep import CRITERIA, LABEL
    ref = lv["refined"]["criteria"]
    fig, ax = plt.subplots(figsize=(C.FULL_W, 2.6))
    fig.subplots_adjust(left=0.36, right=0.97, bottom=0.2, top=0.95)
    y = np.arange(len(CRITERIA))[::-1]
    for yi, n in zip(y, CRITERIA):
        ax.plot([ref[n]["incipient_bvp_wall_E08"]], [yi + 0.12], marker="o", mfc="none", mec=C.COLORS["muted"], ms=5,
                ls="none")
        ax.plot([ref[n]["incipient_wall"]], [yi - 0.12], marker="D", color=C.COLORS["teal"], ms=4, ls="none")
    ax.plot([], [], marker="o", mfc="none", mec=C.COLORS["muted"], ls="none", label="E08: BVP wall states")
    ax.plot([], [], marker="D", color=C.COLORS["teal"], ls="none", label="E08b: E07b forward-net wall states")
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[n] for n in CRITERIA], fontsize=6.8)
    ax.set_xlabel("incipient-separation p$_0$ / p$_a$, 48/64 wall exit state (–)")
    ax.grid(True, axis="x")
    ax.legend(loc="lower right", fontsize=6.5)
    return [C.save_figure(fig, run.out_dir, "e08b_separation_forward")]


if __name__ == "__main__":
    main()
