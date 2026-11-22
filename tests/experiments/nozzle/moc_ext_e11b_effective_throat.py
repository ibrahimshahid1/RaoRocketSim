"""E11b (DIAGNOSTIC): displaced-wall replay with an effective throat and a C2 onset, on the E07b harness.

Opt-in experiment script (moc_extensions_plan_2026-11-21). delta* is the REPOSITORY_HEURISTIC screening estimate
(illustrative); the onset regularisation is a numerical choice, not boundary-layer physics. E11 keeps its record.
Run: .venv-jax/bin/python -m tests.experiments.nozzle.moc_ext_e11b_effective_throat
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np

from scripts import moc_ext_e00_common as C

SCALES = (0.0, 1.0, 2.0)
RT_M, PC, TC, R_GAS = 0.020, 2.0e6, 3500.0, 350.0
ONSET_L = 0.5
RD0, RU0 = 0.382, 1.5


def smoothstep5(xi):
    xi = np.clip(xi, 0.0, 1.0)
    return xi ** 3 * (10.0 - 15.0 * xi + 6.0 * xi * xi)


def main():
    from lrekit.components.nozzle.solvers.nasa import ArcWall
    from lrekit.components.cooling.screens import boundary_layer_displacement
    from tests.experiments.nozzle.moc_ext_e07_forward_replay import bell_nodes_from_solution
    from tests.experiments.nozzle.moc_ext_e07b_exit_limited_replay import replay_wall_limited

    run = C.Run("E11b", __file__)
    e07b = C.OUT_ROOT / "e07b" / "result.json"
    run.add_input(e07b)
    hv_ok = json.load(open(e07b))["harness_verification"]["passes"]
    if not hv_ok:
        run.finish({"status": "SKIPPED-CONDITION", "reason": "E07b harness verification did not pass"})
        return
    report = C.load_report()
    run.add_input(C.PKL["refined"], report["artifact_sha256"]["refined_solution.pkl"])
    sol = C.load_solution("refined")
    kernel = sol.construction_diagnostics["bde_artifacts"]["kernel"]
    theta_B, theta_E = float(kernel.theta_B), float(sol.theta_E)
    arc0 = ArcWall(1.0, RD0, theta_B)
    xb, rb, _ = bell_nodes_from_solution(sol, arc0)
    wc = sol.construction_diagnostics["bde_artifacts"]["bde_region"].wall_contour
    th = np.concatenate([[theta_B], [p.theta for p in wc]])
    wr = np.asarray(sol.wall_raw, dtype=float)
    del sol

    contour = {"x": wr[:, 0] * RT_M, "y": wr[:, 1] * RT_M, "Rt": RT_M, "epsilon": float(report["refined"]["config"]["epsilon"])}
    bl = boundary_layer_displacement(contour, PC, SimpleNamespace(gamma=C.GAMMA, Tc=TC, R_gas=R_GAS))
    ds = np.asarray(bl["delta_star"]) / RT_M
    throat_idx = int(np.argmin(np.abs(wr[:, 1] - 1.0)))
    ds_t = float(np.interp(arc0.x_end, wr[:, 0], ds))
    ds_knot = np.interp(xb, wr[:, 0], ds)
    w = smoothstep5((xb - arc0.x_end) / ONSET_L)
    ds_m = ds_t + w * (ds_knot - ds_t)

    e07b_ref = np.load(C.OUT_ROOT / "e07b" / "forward_wall_refined.npz")
    run.add_input(C.OUT_ROOT / "e07b" / "forward_wall_refined.npz")
    cf_e07b = json.load(open(e07b))["metrics"]["levels"]["refined"]["cf_forward"]

    runs, walls = {}, {}
    for s in SCALES:
        d = s * ds_t
        Rt_e, Rd_e, Ru_e = 1.0 - d, RD0 + d, RU0 + d
        arc = ArcWall(Rt_e, Rd_e, theta_B)
        xk = xb[1:]
        rk = rb[1:] - s * ds_m[1:] / np.cos(th[1:])
        if xk[0] <= arc.x_end:
            raise RuntimeError("first bell knot is upstream of the effective arc end")
        xe = np.concatenate([[arc.x_end], xk])
        re = np.concatenate([[arc.r_end], rk])
        walls[s] = (xe, re)
        rep = replay_wall_limited(theta_B, xe, re, theta_E, n_kernel=64, dtheta_deg=0.03125, Rt=Rt_e, Rd=Rd_e, Ru=Ru_e)
        wall = rep["wall"]
        rec = {"delta_t_over_rt": d, "Rt_eff": Rt_e, "Rd_eff": Rd_e, "Ru_eff": Ru_e,
               "arc_end_eff": [arc.x_end, arc.r_end], "complete": rep["complete"], "stop_reason": rep["stop_reason"],
               "n_rows": rep["n_rows"], "fold_location": rep["fold_location"],
               "fold_inside_nozzle": bool(rep["fold_location"] is not None and rep["fold_location"][0] <= 2.94),
               "cf_geometric_throat": rep.get("cf"),
               "cf_effective_throat": rep["cf"] / Rt_e ** 2 if rep.get("cf") is not None else None,
               "mdot_start": rep["mdot_start"], "mdot_exit": rep.get("mdot_exit"),
               "mdot_exit_rel_to_start": rep.get("mdot_exit_rel_to_start"),
               "limited_stats": rep["limited_stats"], "exit_radius_eff": float(re[-1]),
               "max_offset_over_rw": float(np.max((np.interp(xk, xb, rb) - rk) / np.interp(xk, xb, rb))),
               "wall_exit_M": float(wall["M"][-1]) if wall["M"].size else None}
        runs[s] = rec
        np.savez_compressed(run.out_dir / f"wall_s{s:g}.npz", x=wall["x"], r=wall["r"], M=wall["M"], theta=wall["theta"],
                            knots_x=xe, knots_r=re)
        print(s, rec["stop_reason"], rec["n_rows"], rec["fold_location"], rec["cf_geometric_throat"], rec["mdot_exit_rel_to_start"])
        if rec["fold_inside_nozzle"]:
            print("fold inside nozzle: stop")
            break

    s0 = runs.get(0.0)
    d0 = np.load(run.out_dir / "wall_s0.npz")
    s0_identical = bool(s0 and s0["complete"] and d0["x"].shape == e07b_ref["x"].shape
                        and np.array_equal(d0["x"], e07b_ref["x"]) and np.array_equal(d0["M"], e07b_ref["M"])
                        and s0["cf_geometric_throat"] == cf_e07b)
    done = [s for s in SCALES if s in runs]
    all_complete = len(done) == len(SCALES) and all(runs[s]["complete"] for s in SCALES)
    monotone = None
    if all_complete:
        dcf = np.diff([runs[s]["cf_geometric_throat"] for s in SCALES])
        monotone = bool(np.all(dcf < 0) or np.all(dcf > 0))
    status = "PASS-FEASIBILITY" if (all_complete and monotone and s0_identical) else "FAIL-FEASIBILITY"
    figs = make_figure(run, runs, walls, wr, ds, xb, rb, ds_m)
    result = {
        "status": status,
        "labels": ["DIAGNOSTIC", "onset regularisation is a numerical choice, not boundary-layer physics",
                   "delta* is the REPOSITORY_HEURISTIC estimate (illustrative)"],
        "evidence_class": "neither (feasibility of a coupling architecture)",
        "configuration": {
            "delta_star": "lrekit.components.cooling.screens.boundary_layer_displacement on wall_raw of the 48/64 solution including the "
                          "throat arc, scaled to Rt 20 mm, Pc 2 MPa, gamma 1.23, Tc 3500 K, R 350 J/kg/K, wall 900 K "
                          f"(default); s measured from the geometric throat (wall_raw index {throat_idx}, x = "
                          f"{wr[throat_idx, 0]:g})",
            "effective_throat": "delta*_t = delta*(x_B); arc displaced concentrically: Rt_eff = Rt - s delta*_t, "
                                "Rd_eff = Rd + s delta*_t, KL curvature Ru_eff = Ru + s delta*_t, theta_B unchanged; "
                                "start line and kernel rebuilt at the effective throat (nickerson1985 BLW, PDF 396 / "
                                "printed 5-160, per Opus A A3)",
            "bell": f"r_eff = r_w - s delta*_m / cos(theta_w), delta*_m = delta*_t + w(xi) (delta* - delta*_t), quintic "
                    f"smoothstep w, xi = (x - x_B)/{ONSET_L} Rt; knots = BVP bell knots after B, first knot = effective "
                    "arc end; SplineWall slope_start tan(theta_B), slope_end tan(theta_E)",
            "harness": "E07b replay_wall_limited (exit-plane-limited, script-local), n_kernel 64, dtheta 0.03125 deg",
            "scales": list(SCALES),
            "cf": "exit-plane momentum + pressure / (p0 pi Rt^2) with the geometric Rt (primary) and with Rt_eff",
            "status_rule": "PASS-FEASIBILITY if all scales complete, CF(geometric) strictly monotone in s and s = 0 "
                           "reproduces E07b exactly",
        },
        "metrics": {"delta_star_t_over_rt": ds_t, "delta_star_exit_over_rt": float(ds[-1]),
                    "max_delta_star_over_rw": float(np.max(ds / wr[:, 1])),
                    "s0_reproduces_E07b_exactly": s0_identical, "cf_monotone_in_s": monotone,
                    "runs": {f"{s:g}": v for s, v in runs.items()}},
        "hypotheses": {"H_second_pass_architecture_feasible": "supported" if status == "PASS-FEASIBILITY"
                       else "not supported"},
        "figures": figs,
        "caption_facts": {
            "configuration": ("Exit-plane-limited forward replay (E07b harness) of the 48/64 Nozzle-B wall displaced by "
                              "s x a heuristic delta* (Rt 20 mm, Pc 2 MPa, gamma 1.23) with a concentric effective throat "
                              "and a C2 onset over 0.5 Rt after B, s = 0, 1, 2."),
            "evidence_class": "neither (feasibility of a coupling architecture); DIAGNOSTIC",
            "not_established": "Any boundary-layer physics or loss magnitude; Reshotko-Tucker or Bartz integral results.",
        },
        "not_established": "Any boundary-layer physics or loss magnitude; Reshotko-Tucker or Bartz integral results.",
    }
    run.finish(result, extra_scripts=[C.REPO / "tests/experiments/nozzle/moc_ext_e07b_exit_limited_replay.py",
                                      C.REPO / "tests/experiments/nozzle/moc_ext_e07_forward_replay.py",
                                      C.REPO / "tests/experiments/nozzle/moc_ext_e05_source_flow_oracle.py"])
    print("E11b", status, "s0 identical", s0_identical, "monotone", monotone)


def make_figure(run, runs, walls, wr, ds, xb, rb, ds_m):
    plt = C.setup_style()
    from matplotlib.gridspec import GridSpec
    from lrekit.core.gas_dynamics import isentropic_pressure_ratio
    cols = {0.0: C.COLORS["ink"], 1.0: C.COLORS["teal"], 2.0: C.COLORS["accent"]}
    fig = plt.figure(figsize=(C.FULL_W, 5.0))
    gs = GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.36, left=0.1, right=0.98, top=0.95, bottom=0.1)
    ax = fig.add_subplot(gs[0, 0])
    ss = [s for s in SCALES if s in runs and runs[s]["cf_geometric_throat"] is not None]
    if ss:
        cf0 = runs[0.0]["cf_geometric_throat"] if 0.0 in runs and runs[0.0]["cf_geometric_throat"] else None
        ax.plot(ss, [runs[s]["cf_geometric_throat"] - cf0 for s in ss], color=C.COLORS["ink"], marker="o", ms=4,
                label="geometric R$_t$")
        ax.plot(ss, [runs[s]["cf_effective_throat"] - runs[0.0]["cf_effective_throat"] for s in ss],
                color=C.COLORS["teal"], marker="s", ms=4, ls="--", label="effective R$_{t,eff}$")
    for s_ in SCALES:
        if s_ in runs and runs[s_]["fold_location"] is not None:
            fx, fr = runs[s_]["fold_location"]
            ax.annotate(f"s = {s_:g}: fold at\n(x, r) = ({fx:.2f}, {fr:.2f}) R$_t$", (s_, 0.0), xytext=(-4, -30),
                        textcoords="offset points", ha="right", fontsize=6.3, color=C.COLORS["accent"])
            ax.plot([s_], [0.0], marker="x", color=C.COLORS["accent"], ms=6, ls="none")
    ax.axhline(0, color=C.COLORS["muted"], lw=0.5)
    ax.set_xticks(list(SCALES))
    ax.set_xlabel("displacement scale s (–)")
    ax.set_ylabel("ΔC$_F$ vs s = 0 (–)")
    ax.grid(True)
    ax.legend(loc="best", fontsize=6.5)
    C.panel_label(ax, "a")
    ax = fig.add_subplot(gs[0, 1])
    for s in SCALES:
        p = run.out_dir / f"wall_s{s:g}.npz"
        if p.exists():
            d = np.load(p)
            pr = np.array([isentropic_pressure_ratio(m, C.GAMMA) for m in d["M"]])
            ax.plot(d["x"], pr, color=cols[s], lw=1.0, label=f"s = {s:g}")
    ax.set_yscale("log")
    ax.set_xlabel("x / R$_t$ (–)")
    ax.set_ylabel("wall p / p$_0$ (–)")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=6.5)
    C.panel_label(ax, "b")
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(wr[:, 0], ds / wr[:, 1], color=C.COLORS["teal"], lw=1.1, label="δ*/r$_w$ (s = 1, from the throat)")
    ax.plot(xb, ds_m / rb, color=C.COLORS["accent"], lw=1.0, ls="--", label="blended δ*$_m$/r$_w$ on the bell")
    ax.axvline(xb[0], color=C.COLORS["muted"], lw=0.6, ls=":")
    ax.axvline(xb[0] + ONSET_L, color=C.COLORS["muted"], lw=0.6, ls=":")
    ax.set_xlabel("x / R$_t$ (–)")
    ax.set_ylabel("δ* / r$_w$ (–)")
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=6.3)
    C.panel_label(ax, "c")
    ax = fig.add_subplot(gs[1, 1])
    for s in SCALES:
        if s in walls and s > 0:
            xe, re = walls[s]
            ax.plot(xe, np.interp(xe, xb, rb) - re, color=cols[s], lw=1.0, marker="o", ms=1.8, label=f"s = {s:g}")
    ax.set_xlabel("x / R$_t$ (–)")
    ax.set_ylabel("inward offset at fixed x (R$_t$)")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=6.5)
    C.panel_label(ax, "d")
    return [C.save_figure(fig, run.out_dir, "e11b_effective_throat")]


if __name__ == "__main__":
    main()
