"""Print versions of the B1 figures e07b, e09 and e09c (6.8 in wide, >= 8 pt), re-plotted from saved arrays only.

Opt-in script (moc_extensions_plan_2026-11-21). No solves; numbers and statuses unchanged. Each experiment's
result.json gains a "print_figures" entry.
Run: .venv-jax/bin/python -m tests.experiments.nozzle.moc_ext_print_figures_b1
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

from scripts import moc_ext_e00_common as C

W = 6.8
FS = 8
KL = {"V0": "KL-V0", "V1": "KL-V1", "V2": "KL-V2", "V3": "KL-V3"}


def style():
    plt = C.setup_style()
    plt.rcParams.update({"font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
                         "legend.fontsize": FS, "axes.titlesize": FS, "legend.handlelength": 1.6,
                         "legend.borderaxespad": 0.3, "legend.labelspacing": 0.25})
    return plt


def label(ax, letter):
    ax.text(0.0, 1.03, f"({letter})", transform=ax.transAxes, ha="left", va="bottom", fontsize=8.5,
            fontweight="bold", color=C.COLORS["ink"])


def record(exp_dir: Path, fig: dict, panels: list, inputs: list):
    p = exp_dir / "result.json"
    r = json.load(open(p))
    r["print_figures"] = {
        "script": C.rel(Path(__file__)), "script_sha256": C.sha256(Path(__file__)),
        "command": " ".join([os.path.relpath(sys.executable, C.REPO)] + sys.orig_argv[1:]),
        "git_head": C.git_head(), "inputs_sha256": {C.rel(q): C.sha256(q) for q in inputs},
        "figure": fig, "panels": panels,
        "note": "print re-plot from saved arrays (thresholds, bands and C_D read from this result.json's existing "
                "fields); no numbers, statuses or original figures changed",
    }
    with open(p, "w") as f:
        json.dump(r, f, indent=1)


def fig_e07b(plt):
    from matplotlib.gridspec import GridSpec
    d_dir = C.OUT_ROOT / "e07b"
    res = json.load(open(d_dir / "result.json"))
    th = res["configuration"]["diagnostic_thresholds"]
    data = {l: dict(np.load(d_dir / f"replay_{l}.npz")) for l in ("baseline", "refined")}
    col = {"baseline": C.COLORS["teal"], "refined": C.COLORS["accent"]}
    lab = {"baseline": "32/48", "refined": "48/64"}
    fig = plt.figure(figsize=(W, 2.6))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.2, 1.0, 1.0], wspace=0.5, left=0.08, right=0.99,
                  top=0.9, bottom=0.17)
    ax = fig.add_subplot(gs[0])
    for l, d in data.items():
        ax.plot(d["fwd_wall_x"], d["fwd_wall_M"], color=col[l], lw=1.1, label=f"forward {lab[l]}")
        ax.plot(d["ref_wall_x"][::2], d["ref_wall_M"][::2], ls="none", marker="o", ms=3, mfc="none", mec=col[l],
                mew=0.7, label=f"BVP {lab[l]}")
    ax.set_xlim(0, 3.0)
    ax.set_xlabel("x / R$_t$ (–)")
    ax.set_ylabel("wall Mach number (–)")
    ax.grid(True)
    ax.legend(loc="lower right", ncol=1, frameon=False)
    label(ax, "a")
    for k, (key, lim, scale, yl, letter, unit) in enumerate((
            ("cmp_dM", th["mach_max_abs"], 1e3, "wall ΔM × 10$^3$, forward − BVP (–)", "b", ""),
            ("cmp_dtheta_deg", th["theta_max_abs_deg"], 1.0, "wall Δθ, forward − BVP (deg)", "c", " deg"))):
        ax = fig.add_subplot(gs[k + 1])
        for l, d in data.items():
            ax.plot(d["cmp_x"], d[key] * scale, color=col[l], marker="o", ms=2.2, lw=0.9, label=lab[l])
        ax.axhline(0, color=C.COLORS["muted"], lw=0.5)
        lo, hi = ax.get_ylim()
        span = max(abs(lo), abs(hi))
        ax.set_ylim(-1.15 * span, 1.6 * span)
        ax.text(0.03, 0.97, f"threshold ±{lim:g}{unit}\n(off scale)", transform=ax.transAxes, ha="left", va="top",
                fontsize=FS, color=C.COLORS["muted"])
        ax.set_xlabel("x / R$_t$ (–)")
        ax.set_ylabel(yl)
        ax.grid(True)
        ax.legend(loc="lower left", ncol=2, frameon=False, columnspacing=0.8)
        label(ax, letter)
    out = C.OUT_ROOT / "e07b" / "print"
    figrec = C.save_figure(fig, out, "e07b_print")
    panels = ["(a) wall Mach number vs x/Rt: forward exit-limited net (lines) and BVP BDE wall nodes (circles, every "
              "second node), 32/48 and 48/64",
              "(b) wall dM forward - BVP vs x/Rt at the BVP wall nodes, both levels (diagnostic threshold 0.005 off scale)",
              "(c) wall dtheta (deg) forward - BVP vs x/Rt at the BVP wall nodes, both levels (threshold 0.05 deg off scale)"]
    record(C.OUT_ROOT / "e07b", figrec, panels, [d_dir / "replay_baseline.npz", d_dir / "replay_refined.npz"])
    return figrec, panels


def fig_e09(plt):
    from matplotlib.gridspec import GridSpec
    d_dir = C.OUT_ROOT / "e09"
    res = json.load(open(d_dir / "result.json"))
    sr = json.load(open(d_dir / "series_residuals.json"))
    a = dict(np.load(d_dir / "e09_arrays.npz"))
    col = {"V0": C.COLORS["ink"], "V1": C.COLORS["teal"], "V2": C.COLORS["accent"], "V3": C.COLORS["muted"]}
    ls = {"V0": "-", "V1": "-", "V2": "--", "V3": ":"}
    fig = plt.figure(figsize=(W, 3.6))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.15, 1.0, 1.0], hspace=0.62, wspace=0.55, left=0.09, right=0.99,
                  top=0.93, bottom=0.13)

    ax = fig.add_subplot(gs[0, 0])
    g = "1.23"
    Rc = 1.0 / np.asarray(sr[g]["V0"]["eps"]) - 1.0
    for v in ("V0", "V1"):
        rows = sr[g][v]["rows"]
        ax.plot(Rc, [r["irrot_rms"] for r in rows], color=col[v], marker="o", ms=2.8, lw=1.0)
        ax.plot(Rc, [r["cont_rms"] for r in rows], color=col[v], marker="s", ms=2.6, lw=1.0, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_xticklabels(["1", "2", "4", "8", "16"])
    ax.minorticks_off()
    ax.set_xlabel("R$_c$ / R$_t$ (–)")
    ax.set_ylabel("residual RMS (–)")
    ax.grid(True, which="major")
    ax.set_ylim(1e-11, 3.0)
    ax.set_yticks([1e-10, 1e-7, 1e-4, 1e-1])
    txt = [("order p (— irrot., -- contin.)", C.COLORS["muted"])]
    txt += [(f"{KL[v]}: {sr[g][v]['slope_irrot_rms']:.2f} / {sr[g][v]['slope_cont_rms']:.2f}", col[v])
            for v in ("V0", "V1")]
    for i, (t, c) in enumerate(txt):
        ax.text(0.03, 0.03 + 0.1 * (len(txt) - 1 - i), t, transform=ax.transAxes, fontsize=FS, color=c,
                ha="left", va="bottom")
    label(ax, "a")

    for k, (key, yl, letter, deg) in enumerate((("theta", "start-line Δθ vs KL-V0 (deg)", "b", True),
                                               ("M", "start-line ΔM vs KL-V0 (–)", "c", False))):
        ax = fig.add_subplot(gs[0, k + 1])
        base = a[f"start_nozzle_b_V0_{key}"]
        for v in ("V1", "V2", "V3"):
            y = a[f"start_nozzle_b_{v}_{key}"] - base
            ax.plot(a[f"start_nozzle_b_{v}_r"], np.degrees(y) if deg else y, color=col[v], ls=ls[v], lw=1.1,
                    label=KL[v])
        ax.axhline(0, color=col["V0"], lw=0.7)
        ax.set_xlabel("start-line r / R$_t$ (–)")
        ax.set_ylabel(yl)
        ax.grid(True)
        if letter == "b":
            ax.legend(loc="lower left", frameon=False, ncol=1)
        label(ax, letter)

    ax = fig.add_subplot(gs[1, 0])
    sl = res["metrics"]["start_lines"]
    vs = ["V0", "V1", "V2", "V3"]
    for j, v in enumerate(vs):
        ax.plot(j - 0.1, sl["nozzle_b"][v]["mdot_over_ideal_1d"], marker="o", color=col[v], ms=4.5, ls="none")
        ax.plot(j + 0.1, sl["top"][v]["mdot_over_ideal_1d"], marker="s", mfc="none", mec=col[v], ms=4.5, ls="none")
    ax.axhline(1.0, color=C.COLORS["accent"], lw=0.8, ls="--")
    ax.set_xticks(range(4))
    ax.set_xticklabels([KL[v] for v in vs], fontsize=FS, rotation=30, ha="right")
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylabel("start-line C$_D$ (–)")
    ax.plot([], [], marker="o", color=C.COLORS["ink"], ls="none", label="Nozzle-B, n 64")
    ax.plot([], [], marker="s", mfc="none", mec=C.COLORS["ink"], ls="none", label="TOP, n 48")
    ax.set_ylim(0.990, 1.016)
    ax.legend(loc="upper right", ncol=1, frameon=False, handletextpad=0.2, fontsize=FS)
    ax.grid(True, axis="y")
    label(ax, "d")

    ax = fig.add_subplot(gs[1, 1:])
    x0, M0 = a["replay_V0_wall_x"], a["replay_V0_wall_M"]
    x3, M3 = a["replay_V3_wall_x"], a["replay_V3_wall_M"]
    ax.plot(x3, M3 - np.interp(x3, x0, M0), color=col["V3"], ls="-", lw=1.3, label="KL-V3 − KL-V0")
    ax.plot([], [], color=col["V1"], lw=1.1, label="KL-V1: net stopped at RRC 4 (near-axis fold)")
    ax.plot([], [], color=col["V2"], lw=1.1, ls="--", label="KL-V2: no RRC completed")
    band = float(res["metrics"]["t19_refinement_mach_max_abs"])
    ax.axhline(band, color=C.COLORS["teal"], lw=0.8, ls="--")
    ax.text(0.0, band, f" t19 max |ΔM| = {band:.1e}", va="bottom", ha="left", fontsize=FS, color=C.COLORS["teal"])
    ax.set_xlim(0.0, float(x3.max()) * 1.02)
    ax.set_ylim(0.0, 0.07)
    ax.set_xlabel("x / R$_t$ on the throat arc, start line to B (–)")
    ax.set_ylabel("replay wall ΔM (–)")
    ax.grid(True)
    ax.legend(loc="upper left", frameon=False, fontsize=FS)
    label(ax, "e")
    out = C.OUT_ROOT / "e09" / "print"
    figrec = C.save_figure(fig, out, "e09_print")
    panels = [
        "(a) series residual RMS vs Rc/Rt (gamma 1.23; gammas 1.2 and 1.4 coincide visually): irrotationality (solid) "
        "and continuity (dashed) for KL-V0 and KL-V1; p = fitted order in eps = 1/(1 + Rc/Rt) over all five Rc",
        "(b) Nozzle-B start line (Ru/Rt 1.5, gamma 1.23, n = 64): dtheta vs KL-V0 against start-line r/Rt for "
        "KL-V1, KL-V2, KL-V3",
        "(c) same start lines: dM vs KL-V0 against start-line r/Rt",
        "(d) start-line discharge coefficient C_D = mdot/mdot_1D by variant, Nozzle-B (n 64) and TOP (gamma 1.2, "
        "n 48); dashed line C_D = 1",
        "(e) forward-replay (E09, unlimited harness) wall dM KL-V3 - KL-V0 along the throat arc from the start line "
        "to B on the 48/64 wall; KL-V1 and KL-V2 produce no comparable wall states; dashed line = t19 field max |dM|",
    ]
    record(C.OUT_ROOT / "e09", figrec, panels, [d_dir / "e09_arrays.npz", d_dir / "series_residuals.json"])
    return figrec, panels


def fig_e09c(plt):
    from matplotlib.gridspec import GridSpec
    d_dir = C.OUT_ROOT / "e09c"
    res = json.load(open(d_dir / "result.json"))
    a = dict(np.load(d_dir / "e09c_arrays.npz"))
    band = res["configuration"]["t19_band"]
    m = res["metrics"]
    xB = float(m["V0"]["B"]["x"])
    fig = plt.figure(figsize=(W, 2.8))
    gs = GridSpec(1, 3, figure=fig, wspace=0.58, left=0.1, right=0.99, top=0.9, bottom=0.17)
    specs = ((0, "cmp_x", "cmp_dM", band["mach_max_abs"], "wall ΔM, KL-V3 − KL-V0 (–)", "x / R$_t$ (–)", "a",
              "t19 band ±8.0e-4"),
             (1, "cmp_x", "cmp_dp", band["pressure_ratio_max_abs"], "wall Δ(p/p$_0$), KL-V3 − KL-V0 (–)",
              "x / R$_t$ (–)", "b", "t19 band ±7.5e-5"),
             (2, "V0_exit_r", "exit_dM", band["mach_max_abs"], "exit-plane ΔM, KL-V3 − KL-V0 (–)",
              "r / R$_t$ at x = 2.94 R$_t$ (–)", "c", "t19 band ±8.0e-4"))
    for k, xk, yk, b, yl, xl, letter, bl in specs:
        ax = fig.add_subplot(gs[k])
        ax.plot(a[xk], a[yk], color=C.COLORS["accent"], lw=1.2)
        ax.axhline(0, color=C.COLORS["muted"], lw=0.5)
        ax.axhline(b, color=C.COLORS["teal"], lw=0.7, ls="--")
        ax.axhline(-b, color=C.COLORS["teal"], lw=0.7, ls="--")
        if letter in ("a", "b"):
            ax.axvline(xB, color=C.COLORS["muted"], lw=0.6, ls=":")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(True)
        label(ax, letter)
        if letter == "a":
            ax.text(0.97, 0.78, bl + "\n(dashed)", transform=ax.transAxes, ha="right", va="center", fontsize=FS,
                    color=C.COLORS["teal"])
            ax.text(xB, 0.03, " B", transform=ax.get_xaxis_transform(), va="bottom", fontsize=FS, color=C.COLORS["muted"])
        if letter == "b":
            ax.text(0.97, 0.3, bl + "\n(dashed)", transform=ax.transAxes, ha="right", va="center", fontsize=FS,
                    color=C.COLORS["teal"])
        if letter == "c":
            ax.set_ylim(-0.02, 0.17)
            ax.text(0.03, 0.97, f"ΔC$_F$/C$_F$ = {m['dCF_over_CF']:.2e}\n"
                    f"({m['ratio_to_t19_band']['dCF_over_CF']:.0f}× t19 band)\n" + bl + "\n(dashed)",
                    transform=ax.transAxes, ha="left", va="top", fontsize=FS, color=C.COLORS["ink"])
    out = C.OUT_ROOT / "e09c" / "print"
    figrec = C.save_figure(fig, out, "e09c_print")
    panels = [
        "(a) wall dM KL-V3 - KL-V0 vs x/Rt along the forward-replay wall (48/64, E07b exit-limited harness); dotted "
        "line B; dashed lines t19 field band +/-8.04e-4",
        "(b) wall d(p/p0) KL-V3 - KL-V0 vs x/Rt; dashed lines t19 band +/-7.49e-5",
        "(c) exit-plane dM KL-V3 - KL-V0 vs r/Rt at x = 2.94 Rt; dashed lines t19 band +/-8.04e-4; annotation "
        "dCF/CF = -1.22e-2 (307x the t19 band 3.98e-5)",
    ]
    record(C.OUT_ROOT / "e09c", figrec, panels, [d_dir / "e09c_arrays.npz"])
    return figrec, panels


def main():
    plt = style()
    for f in (fig_e07b, fig_e09, fig_e09c):
        rec, panels = f(plt)
        print(rec["pdf"], rec["png"])
        for p in panels:
            print("  ", p)


if __name__ == "__main__":
    main()
