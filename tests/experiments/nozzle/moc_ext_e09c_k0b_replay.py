"""E09c (K0b replay, DIAGNOSTIC): V0 vs V3 KL start line, forward replay on the accepted 48/64 Nozzle-B wall.

Opt-in experiment script (moc_extensions_plan_2026-11-21). E07b exit-limited harness; kliegel_levine monkeypatched
in-process only (E09 local copy). Forward replay on the V0-optimised wall; not a corrected design. No BVP solve.
Run: .venv-jax/bin/python -m tests.experiments.nozzle.moc_ext_e09c_k0b_replay
"""

from __future__ import annotations

import json
import math

import numpy as np

from scripts import moc_ext_e00_common as C

VARS = ("V0", "V3")
N_KERNEL = 64
DTHETA_DEG = 0.03125


def main():
    from lrekit.core.gas_dynamics import isentropic_pressure_ratio
    from lrekit.components.nozzle.solvers.nasa import ArcWall, _make_throat_initial_line
    from tests.experiments.nozzle.moc_ext_e07_forward_replay import bell_nodes_from_solution
    from tests.experiments.nozzle.moc_ext_e07b_exit_limited_replay import replay_wall_limited
    from tests.experiments.nozzle.moc_ext_e09_kl_variants import patched
    from tests.experiments.nozzle.moc_ext_e09b_startline_review import discharge

    run = C.Run("E09c", __file__)
    report = C.load_report()
    band = {"mach_max_abs": float(report["refinement"]["metrics"]["mach_max_abs"]),
            "theta_max_abs_deg": float(report["refinement"]["metrics"]["theta_max_abs_deg"]),
            "pressure_ratio_max_abs": float(report["refinement"]["metrics"]["pressure_ratio_max_abs"]),
            "cf_rel": float(report["refinement"]["metrics"]["thrust_coefficient_relative_delta"])}
    e07b = C.OUT_ROOT / "e07b"
    run.add_input(e07b / "result.json")
    run.add_input(e07b / "forward_wall_refined.npz")
    cf_e07b = json.load(open(e07b / "result.json"))["metrics"]["levels"]["refined"]["cf_forward"]
    ref_wall = np.load(e07b / "forward_wall_refined.npz")
    run.add_input(C.PKL["refined"], report["artifact_sha256"]["refined_solution.pkl"])
    sol = C.load_solution("refined")
    kernel = sol.construction_diagnostics["bde_artifacts"]["kernel"]
    theta_B, theta_E = float(kernel.theta_B), float(sol.theta_E)
    arc = ArcWall(1.0, 0.382, theta_B)
    xb, rb, _ = bell_nodes_from_solution(sol, arc)
    del sol

    reps = {}
    for v in VARS:
        with patched(v):
            start = _make_throat_initial_line(1.0, 1.5, theta_B, C.GAMMA, N_KERNEL, "kliegel_levine")
            rep = replay_wall_limited(theta_B, xb, rb, theta_E, n_kernel=N_KERNEL, dtheta_deg=DTHETA_DEG, keep_net=True)
        net = rep.pop("net")
        kr = next((j for j, row in enumerate(net.rows) if row[0].x >= arc.x_end - 1e-9 or row[0].theta >= theta_B - 1e-6),
                  None)
        rep["kernel_rows_to_B"] = None if kr is None else kr + 1
        rep["reached_theta_B"] = kr is not None
        rep["B"] = ({"x": float(net.rows[kr][0].x), "M": float(net.rows[kr][0].M),
                     "theta_deg": math.degrees(net.rows[kr][0].theta)} if kr is not None else None)
        rep["C_D"] = discharge(start)
        del net
        reps[v] = rep
        print(v, rep["stop_reason"], rep["n_rows"], rep["fold_location"], rep.get("cf"), rep["B"], rep["C_D"])

    v0, v3 = reps["V0"], reps["V3"]
    control = bool(v0["complete"] and np.array_equal(v0["wall"]["x"], ref_wall["x"])
                   and np.array_equal(v0["wall"]["M"], ref_wall["M"]) and v0["cf"] == cf_e07b)
    pv = np.vectorize(lambda m: isentropic_pressure_ratio(float(m), C.GAMMA))
    out = {"control_V0_reproduces_E07b": control}
    arrays = {}
    for v, r in reps.items():
        arrays[f"{v}_wall_x"] = r["wall"]["x"]
        arrays[f"{v}_wall_M"] = r["wall"]["M"]
        arrays[f"{v}_wall_theta"] = r["wall"]["theta"]
        if "exit_plane" in r:
            arrays[f"{v}_exit_r"] = r["exit_plane"]["r"]
            arrays[f"{v}_exit_M"] = r["exit_plane"]["M"]
    fold_inside = bool(v3["fold_location"] is not None and v3["fold_location"][0] <= 2.94)
    if v3["complete"]:
        x0 = v0["wall"]["x"]
        m = (x0 >= v3["wall"]["x"].min()) & (x0 <= v3["wall"]["x"].max())
        xs = x0[m]
        dM = np.interp(xs, v3["wall"]["x"], v3["wall"]["M"]) - v0["wall"]["M"][m]
        dp = pv(np.interp(xs, v3["wall"]["x"], v3["wall"]["M"])) - pv(v0["wall"]["M"][m])
        dth = np.degrees(np.interp(xs, v3["wall"]["x"], v3["wall"]["theta"]) - v0["wall"]["theta"][m])
        bell = xs > arc.x_end
        r0 = v0["exit_plane"]["r"]
        dMe = np.interp(r0, v3["exit_plane"]["r"], v3["exit_plane"]["M"]) - v0["exit_plane"]["M"]
        arrays.update({"cmp_x": xs, "cmp_dM": dM, "cmp_dp": dp, "cmp_dtheta_deg": dth, "exit_dM": dMe})
        cf_rel = (v3["cf"] - v0["cf"]) / v0["cf"]
        def ratio(a, b):
            return a / b if b else None
        out.update({
            "wall_max_abs_dM": float(np.max(np.abs(dM))), "wall_max_abs_dM_bell": float(np.max(np.abs(dM[bell]))),
            "wall_max_abs_dp_over_p0": float(np.max(np.abs(dp))),
            "wall_max_abs_dtheta_deg": float(np.max(np.abs(dth))),
            "wall_exit_dM": float(v3["wall"]["M"][-1] - v0["wall"]["M"][-1]),
            "wall_exit_dp_over_p0": float(pv(v3["wall"]["M"][-1]) - pv(v0["wall"]["M"][-1])),
            "exit_plane_max_abs_dM": float(np.max(np.abs(dMe))),
            "cf_V0": v0["cf"], "cf_V3": v3["cf"], "dCF_over_CF": cf_rel,
            "mdot_exit_V0": v0["mdot_exit"], "mdot_exit_V3": v3["mdot_exit"],
            "mdot_exit_rel_V3_vs_V0": (v3["mdot_exit"] - v0["mdot_exit"]) / v0["mdot_exit"],
            "mdot_exit_rel_to_start": {"V0": v0["mdot_exit_rel_to_start"], "V3": v3["mdot_exit_rel_to_start"]},
            "ratio_to_t19_band": {"wall_dM": ratio(float(np.max(np.abs(dM))), band["mach_max_abs"]),
                                  "wall_dp": ratio(float(np.max(np.abs(dp))), band["pressure_ratio_max_abs"]),
                                  "wall_dtheta": ratio(float(np.max(np.abs(dth))), band["theta_max_abs_deg"]),
                                  "dCF_over_CF": ratio(abs(cf_rel), band["cf_rel"])},
        })
    out.update({v: {"complete": r["complete"], "stop_reason": r["stop_reason"], "n_rows": r["n_rows"],
                    "fold_location": r["fold_location"], "reached_theta_B": r["reached_theta_B"],
                    "kernel_rows_to_B": r["kernel_rows_to_B"], "B": r["B"], "C_D": r["C_D"], "cf": r.get("cf"),
                    "mdot_start": r["mdot_start"], "mdot_exit": r.get("mdot_exit"),
                    "limited_stats": r["limited_stats"]} for v, r in reps.items()})
    if "B" in v3 and v3["B"] and v0["B"]:
        out["M_B_V3_minus_V0"] = v3["B"]["M"] - v0["B"]["M"]
    np.savez_compressed(run.out_dir / "e09c_arrays.npz", **arrays)
    if not control:
        status = "FAIL-DIAGNOSTIC"
        reason = "control: V0 does not reproduce E07b"
    elif not v3["complete"]:
        status = "FAIL-DIAGNOSTIC"
        reason = (f"V3 {v3['stop_reason']} at {v3['fold_location']}"
                  + (" (inside the nozzle)" if fold_inside else ""))
    else:
        status, reason = "PASS-DIAGNOSTIC", "control reproduces E07b; V3 replay complete"
    figs = make_figure(run, arrays, band, out, arc.x_end)
    result = {
        "status": status, "reason": reason,
        "labels": ["DIAGNOSTIC", "forward replay on the V0-optimised wall; not a corrected design",
                   "D-K0-BVP still not approved (D10)"],
        "evidence_class": "neither (sensitivity of a forward replay to the start-line model); control is verification",
        "configuration": {
            "variants": {"V0": "repo KL series, M = hypot(U,V) (production)",
                         "V3": "v3 z*y^3 sign negated and output M from M* (Anderson Eq. 3.37); push basis M (inactive, "
                               "E09b)"},
            "harness": "scripts/moc_ext_e07b_exit_limited_replay.replay_wall_limited, refined 48/64 wall, n_kernel 64, "
                       "0.03125 deg, mass_tol 0.02, on_invalid 'stop', x_exit 2.94; kliegel_levine monkeypatched "
                       "in-process (scripts/moc_ext_e09_kl_variants.patched), production untouched",
            "comparison": "V3 wall states interpolated in x onto the V0 forward wall points; exit plane interpolated in r",
            "t19_band": band,
        },
        "metrics": out,
        "hypotheses": {"V3_changes_exceed_t19_band": (
            "yes" if (v3["complete"] and max(out["ratio_to_t19_band"]["wall_dM"], out["ratio_to_t19_band"]["dCF_over_CF"]) > 1)
            else ("no" if v3["complete"] else "undetermined"))},
        "figures": figs,
        "caption_facts": {
            "configuration": ("Exit-plane-limited forward replay (E07b harness) of the accepted 48/64 Nozzle-B wall "
                              "(gamma 1.23) with the repo KL start line (V0) and the sign-corrected, true-Mach start line "
                              "(V3); differences V3 - V0 against the t19 refinement deltas."),
            "evidence_class": "neither (sensitivity); DIAGNOSTIC; forward replay on the V0-optimised wall, not a corrected design",
            "not_established": "A corrected optimum contour or its CF (needs D-K0-BVP, user decision D10); physical validity.",
        },
        "not_established": "A corrected optimum contour or its CF (needs D-K0-BVP, user decision D10); physical validity.",
    }
    run.finish(result, extra_scripts=[C.REPO / "tests/experiments/nozzle/moc_ext_e07b_exit_limited_replay.py",
                                      C.REPO / "tests/experiments/nozzle/moc_ext_e07_forward_replay.py",
                                      C.REPO / "tests/experiments/nozzle/moc_ext_e09_kl_variants.py",
                                      C.REPO / "tests/experiments/nozzle/moc_ext_e09b_startline_review.py",
                                      C.REPO / "tests/experiments/nozzle/moc_ext_e05_source_flow_oracle.py"])
    print("E09c", status, reason, {k: out.get(k) for k in ("wall_max_abs_dM", "wall_max_abs_dp_over_p0", "dCF_over_CF",
                                                           "M_B_V3_minus_V0", "ratio_to_t19_band")})


def make_figure(run, a, band, out, x_B):
    plt = C.setup_style()
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(C.FULL_W, 4.8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.5, wspace=0.38, left=0.11, right=0.98, top=0.95, bottom=0.1)
    if "cmp_x" not in a:
        ax = fig.add_subplot(gs[:, :])
        for v, c in (("V0", C.COLORS["ink"]), ("V3", C.COLORS["accent"])):
            ax.plot(a[f"{v}_wall_x"], a[f"{v}_wall_M"], color=c, lw=1.0, label=v)
        ax.set_xlabel("x / R$_t$ (–)")
        ax.set_ylabel("wall Mach number (–)")
        ax.legend()
        return [C.save_figure(fig, run.out_dir, "e09c_k0b_replay")]
    specs = ((0, 0, "cmp_dM", band["mach_max_abs"], "wall ΔM, V3 − V0 (–)", "a"),
             (0, 1, "cmp_dp", band["pressure_ratio_max_abs"], "wall Δ(p/p$_0$), V3 − V0 (–)", "b"),
             (1, 0, "cmp_dtheta_deg", band["theta_max_abs_deg"], "wall Δθ, V3 − V0 (deg)", "c"))
    for row, col_, key, b, yl, letter in specs:
        ax = fig.add_subplot(gs[row, col_])
        ax.plot(a["cmp_x"], a[key], color=C.COLORS["accent"], lw=1.0, label="V3 − V0")
        ax.axhspan(-b, b, color=C.COLORS["teal"], alpha=0.15, lw=0, label="t19 refinement max")
        ax.axhline(0, color=C.COLORS["muted"], lw=0.5)
        ax.axvline(x_B, color=C.COLORS["muted"], lw=0.6, ls=":")
        ax.set_xlabel("x / R$_t$ (–)")
        ax.set_ylabel(yl)
        ax.grid(True)
        ax.legend(loc="best", fontsize=6.3)
        C.panel_label(ax, letter)
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(a["V0_exit_r"], a["exit_dM"], color=C.COLORS["accent"], lw=1.0, label="exit-plane ΔM, V3 − V0")
    ax.axhspan(-band["mach_max_abs"], band["mach_max_abs"], color=C.COLORS["teal"], alpha=0.15, lw=0,
               label="t19 refinement max |ΔM|")
    ax.axhline(0, color=C.COLORS["muted"], lw=0.5)
    ax.text(0.02, 0.04, f"ΔC$_F$/C$_F$ = {out['dCF_over_CF']:.2e} ({out['ratio_to_t19_band']['dCF_over_CF']:.0f}× t19 band)",
            transform=ax.transAxes, fontsize=6.5, color=C.COLORS["ink"])
    ax.set_xlabel("r / R$_t$ at x = 2.94 R$_t$ (–)")
    ax.set_ylabel("exit-plane ΔM (–)")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=6.3)
    C.panel_label(ax, "d")
    return [C.save_figure(fig, run.out_dir, "e09c_k0b_replay")]


if __name__ == "__main__":
    main()
