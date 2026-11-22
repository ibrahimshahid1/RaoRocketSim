"""E11: displaced-wall forward replay, feasibility of a TDK-style second inviscid pass with the existing primal.

Opt-in experiment script (moc_extensions_plan_2026-11-21). delta* is the REPOSITORY_HEURISTIC screening estimate
(lrekit.components.cooling.screens.boundary_layer_displacement), used only as an illustrative displacement shape.
Run: .venv-jax/bin/python -m tests.experiments.nozzle.moc_ext_e11_displaced_wall
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from scripts import moc_ext_e00_common as C

SCALES = (0.5, 1.0, 2.0, 4.0)
RT_M = 0.020
PC = 2.0e6
TC = 3500.0
R_GAS = 350.0


def main():
    from lrekit.components.nozzle.solvers.nasa import ArcWall
    from lrekit.components.cooling.screens import boundary_layer_displacement
    from tests.experiments.nozzle.moc_ext_e07_forward_replay import (RD_OVER_RT, RT, bell_nodes_from_solution, diagnose_fold,
                                                    make_wall, replay_wall)

    run = C.Run("E11", __file__)
    report = C.load_report()
    run.add_input(C.PKL["refined"], report["artifact_sha256"]["refined_solution.pkl"])
    sol = C.load_solution("refined")
    kernel = sol.construction_diagnostics["bde_artifacts"]["kernel"]
    theta_B = float(kernel.theta_B)
    theta_E = float(sol.theta_E)
    arc = ArcWall(RT, RD_OVER_RT * RT, theta_B)
    xb, rb, seam = bell_nodes_from_solution(sol, arc)
    wc = sol.construction_diagnostics["bde_artifacts"]["bde_region"].wall_contour
    if len(wc) != len(xb) - 1 or np.max(np.abs(np.array([p.x for p in wc]) - xb[1:])) > 1e-12:
        raise RuntimeError("bell knots do not match the BDE wall nodes")
    th = np.concatenate([[theta_B], [p.theta for p in wc]])

    we = np.asarray(sol.wall_export, dtype=float)
    contour = {"x": we[:, 0] * RT_M, "y": we[:, 1] * RT_M, "Rt": RT_M, "epsilon": float(report["refined"]["config"]["epsilon"])}
    prop = SimpleNamespace(gamma=C.GAMMA, Tc=TC, R_gas=R_GAS)
    bl = boundary_layer_displacement(contour, PC, prop)
    ds_export = np.asarray(bl["delta_star"]) / RT_M
    ds_knot = np.interp(xb, we[:, 0], ds_export)
    ds_over_r = ds_knot / rb

    runs = {}
    walls = {}
    for s in SCALES:
        rb_eff = rb - s * ds_knot / np.cos(th)
        snapped = float(rb_eff[0] - arc.r_end)
        rb_eff[0] = arc.r_end
        walls[s] = rb_eff
        rep = replay_wall(theta_B, xb, rb_eff, theta_E, n_kernel=64, dtheta_deg=0.03125, keep_net=True)
        net = rep.pop("net")
        fd = None
        if not net.complete:
            _, _, wall = make_wall(theta_B, xb, rb_eff, theta_E)
            fd = diagnose_fold(net, wall, C.GAMMA, 0.03125)
            fd.pop("rev_xy", None)
            fd.pop("attempted_row_xy", None)
        del net
        runs[s] = {
            "fold_diagnosis": fd,
            "complete": rep["complete"], "stop_reason": rep["stop_reason"], "n_rows": rep["n_rows"],
            "fold_location": rep["fold_location"], "cf": rep.get("cf"),
            "B_knot_offset_snapped": snapped, "exit_radius_eff": float(rb_eff[-1]),
            "epsilon_eff": float(rb_eff[-1] ** 2), "max_inward_offset_over_rt": float(np.max(rb - rb_eff)),
            "max_offset_over_rw": float(np.max((rb - rb_eff) / rb)),
            "forward_wall_x_max": float(rep["wall"]["x"].max()) if rep["wall"]["x"].size else None,
            "march_wall_time_s": rep["march_wall_time_s"],
        }
        print(s, runs[s]["stop_reason"], runs[s]["n_rows"], runs[s]["fold_location"], runs[s]["cf"])
    del sol

    cfs = [runs[s]["cf"] for s in SCALES]
    all_complete = all(runs[s]["complete"] for s in SCALES)
    monotone = None
    if all_complete:
        d = np.diff(np.array(cfs))
        monotone = bool(np.all(d < 0) or np.all(d > 0))
    status = "PASS-FEASIBILITY" if (all_complete and monotone) else "FAIL-FEASIBILITY"
    np.savez_compressed(run.out_dir / "e11_arrays.npz", xb=xb, rb=rb, ds_knot=ds_knot, export_x=we[:, 0],
                        export_r=we[:, 1], ds_export=ds_export, **{f"rb_eff_{s}": walls[s] for s in SCALES})
    figs = make_figure(run, xb, rb, ds_knot, walls, runs, we, ds_export)
    result = {
        "status": status,
        "evidence_class": "neither (feasibility of a coupling architecture)",
        "configuration": {
            "delta_star": ("lrekit.components.cooling.screens.boundary_layer_displacement on the 48/64 wall_export scaled to Rt = 20 mm, "
                           "Pc = 2 MPa, gamma 1.23, Tc 3500 K, R 350 J/kg/K, default wall temperature 900 K; "
                           "REPOSITORY_HEURISTIC, illustrative"),
            "delta_star_origin": ("the function puts delta* = 0 at the contour point nearest r = Rt; wall_export starts at "
                                  f"x = {we[0, 0]:.6f} Rt (just downstream of B = {arc.x_end:.6f} Rt), so that point is "
                                  "the origin; knots are linearly interpolated from the export stations"),
            "displaced_wall": "r_eff = r_w - s*delta*/cos(theta_w) at the BVP bell knots (theta_w from the BDE wall nodes, "
                              "theta_B at B); B knot held on the unchanged arc end; SplineWall slope_start tan(theta_B), "
                              "slope_end tan(theta_E) as in E07",
            "scales": list(SCALES), "replay": "E07 replay_wall, n_kernel 64, dtheta 0.03125 deg, x_exit 2.94",
        },
        "metrics": {
            "delta_star_over_rt_exit": float(ds_export[-1]), "delta_star_over_rt_max": float(ds_export.max()),
            "max_delta_star_over_rw": float(np.max(ds_over_r)),
            "delta_star_over_rt_at_B": float(ds_knot[0]),
            "bl_model": bl["model"], "bl_basis": bl["correlation_basis"],
            "runs": {str(s): v for s, v in runs.items()},
            "cf_monotone_in_s": monotone,
        },
        "hypotheses": {"H_forward_net_completes_and_cf_monotone": "supported" if status == "PASS-FEASIBILITY"
                       else "not supported"},
        "figures": figs,
        "caption_facts": {
            "configuration": ("Forward bounded-net replay of the 48/64 Nozzle-B wall displaced inward by s x a heuristic "
                              "delta*(x) (Rt 20 mm, Pc 2 MPa, gamma 1.23), s = 0.5-4, throat arc and KL start line "
                              "unchanged. (a) fold station of each replay (no CF: every net stopped); (b) inward wall "
                              "offset per s; (c) delta*/r_w."),
            "evidence_class": "neither (feasibility of a coupling architecture)",
            "not_established": "Any boundary-layer physics or loss magnitude; Reshotko-Tucker or Bartz integral results.",
        },
        "not_established": "Any boundary-layer physics or loss magnitude; Reshotko-Tucker or Bartz integral results.",
        "notes": [
            "Spec figure (b) asks for wall p/p0 vs x per s; no displaced-wall pressure exists because every replay stops "
            "at the first post-kernel RRC, so panel (b) shows the replayed wall offsets instead.",
            "Unlike E07 (undisplaced wall, fold only downstream of the exit plane), every displaced wall folds on the "
            "first post-kernel RRC inside the nozzle near r = 1.03-1.04 Rt, moving upstream as s grows: the heuristic "
            "delta* grows like (x - x0)^0.8 from an origin just downstream of B, so the displaced bell turns inward "
            "immediately after B and its first compression RRC crosses the last kernel RRC.",
        ],
    }
    e07 = C.OUT_ROOT / "e07" / "level_refined.json"
    if e07.exists():
        run.add_input(e07)
    run.finish(result, extra_scripts=[C.REPO / "tests/experiments/nozzle/moc_ext_e07_forward_replay.py"])
    print("E11", status)


def make_figure(run, xb, rb, ds_knot, walls, runs, we, ds_export):
    plt = C.setup_style()
    from matplotlib.gridspec import GridSpec
    cols = {0.5: C.COLORS["muted"], 1.0: C.COLORS["teal"], 2.0: C.COLORS["ink"], 4.0: C.COLORS["accent"]}
    fig = plt.figure(figsize=(C.FULL_W, 4.8))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.0], hspace=0.5, wspace=0.34,
                  left=0.1, right=0.98, top=0.95, bottom=0.1)
    ax = fig.add_subplot(gs[0, 0])
    s_arr = np.array(SCALES)
    cfs = [runs[s]["cf"] for s in SCALES]
    if all(c is not None for c in cfs):
        ax.plot(s_arr, cfs, color=C.COLORS["ink"], marker="o", ms=3.5)
        ax.set_ylabel("C$_F$ forward net (–)")
    else:
        fx = [runs[s]["fold_location"][0] if runs[s]["fold_location"] else np.nan for s in SCALES]
        ax.plot(s_arr, fx, color=C.COLORS["accent"], marker="x", ms=4.5, ls="none", label="fold station")
        ax.axhline(2.94, color=C.COLORS["muted"], lw=0.7, ls=":")
        ax.text(s_arr[0], 2.94, " exit plane", va="bottom", fontsize=6.5, color=C.COLORS["muted"])
        e07 = C.OUT_ROOT / "e07" / "level_refined.json"
        if e07.exists():
            import json
            fx0 = json.load(open(e07))["fold_location"][0]
            ax.axhline(fx0, color=C.COLORS["teal"], lw=0.8, ls="--", label=f"s = 0 (E07): x = {fx0:.2f}")
        ax.set_ylim(0, 6.0)
        ax.set_ylabel("fold x / R$_t$ (no C$_F$) (–)")
        ax.legend(loc="upper right", fontsize=6.3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(s_arr)
    ax.set_xticklabels([f"{s:g}" for s in SCALES])
    ax.minorticks_off()
    ax.set_xlabel("displacement scale s (–)")
    ax.grid(True)
    C.panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    for s in SCALES:
        ax.plot(xb, rb - walls[s], color=cols[s], marker="o", ms=1.8, lw=0.9, label=f"s = {s:g}")
    ax.set_xlabel("x / R$_t$ (–)")
    ax.set_ylabel("inward offset r$_w$ − r$_{eff}$ (R$_t$)")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=6.5)
    C.panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, :])
    ax.plot(we[:, 0], ds_export / we[:, 1], color=C.COLORS["teal"], lw=1.1, label="δ*/r$_w$ at export stations (s = 1)")
    ax.plot(xb, ds_knot / rb, ls="none", marker="o", ms=2.2, mfc="none", mec=C.COLORS["ink"], label="bell knots")
    ax.set_xlabel("x / R$_t$ (–)")
    ax.set_ylabel("δ* / r$_w$ (–)")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=6.5)
    C.panel_label(ax, "c")
    return [C.save_figure(fig, run.out_dir, "e11_displaced_wall")]


if __name__ == "__main__":
    main()
