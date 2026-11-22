"""Print version of the E12 figure, re-plotted from saved data only (no JAX runs).

Inputs: opus_b/e12/e12_sweep.npz, opus_b/e12/result.json (pre-registered Taylor data),
opus_b/e12b/result.json (event classification; POST-HOC J1 forward/backward).
Writes opus_b/e12/print/ and records the outputs under result.json["print_figures"]
without changing any number or status.

.venv-jax/bin/python tests/experiments/nozzle/moc_ext_print_figures_b3.py
"""
from __future__ import annotations

from lrekit.paths import repository_root as _repository_root

import hashlib
import json
import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = str(_repository_root())
B = os.path.join(ROOT, "builds", "moc_extensions_plan_2026-11-21", "opus_b")
E12, E12B = os.path.join(B, "e12"), os.path.join(B, "e12b")
OUT = os.path.join(E12, "print")
INK, TEAL, MUTED, ACCENT = "#19333C", "#176D76", "#51616B", "#C26A2E"
NAMES = ["J1_mdot_BD", "J2_M_B", "J3_x_D_axis"]
LABELS = ["$J_1$", "$J_2$", "$J_3$"]
COLORS = [INK, TEAL, ACCENT]
H_DEG = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
NOISE_FACTOR = 100.0
E12B_POINT = 1


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def git_head():
    gd = os.path.join(ROOT, ".git")
    head = open(os.path.join(gd, "HEAD")).read().strip()
    if not head.startswith("ref:"):
        return head
    ref = head.split(None, 1)[1]
    p = os.path.join(gd, ref)
    if os.path.exists(p):
        return open(p).read().strip()
    for line in open(os.path.join(gd, "packed-refs")):
        if line.strip().endswith(ref):
            return line.split()[0]
    return "unavailable"


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    npz_path = os.path.join(E12, "e12_sweep.npz")
    res_path = os.path.join(E12, "result.json")
    b_path = os.path.join(E12B, "result.json")
    res = json.load(open(res_path))
    res_sha_before = res.get("print_figures", {}).get("pre_amendment_result_sha256") or sha(res_path)
    sw = np.load(npz_path)
    rb = json.load(open(b_path))
    th = sw["theta_B_deg"]
    dJ = sw["dJ_dthetaB_per_rad"]
    pc = sw["pattern_change"]
    valid = sw["reached_wall"]

    def idx(v):
        return int(np.argmin(np.abs(th - v)))
    ev = set()
    for iv in rb["event_free_intervals"]:
        ev.add(idx(iv["lo_deg"]) - 1)
        ev.add(idx(iv["hi_deg"]))
    row_ev = sorted(k for k in ev if pc[k])
    clamp_ev = sorted(k for k in ev if not pc[k])

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "legend.fontsize": 8, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "text.color": INK, "axes.linewidth": 0.7, "pdf.fonttype": 42})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.6), gridspec_kw=dict(width_ratios=[1.0, 1.0], wspace=0.36))
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.17, top=0.75)

    y = dJ[:, 2] * np.pi / 180.0
    coef = np.polyfit(th[valid], y[valid], 1)
    yd = (y - np.polyval(coef, th)) * 1e3
    for k in row_ev:
        ax1.axvline(0.5 * (th[k] + th[k + 1]), color=ACCENT, lw=0.8, ls="-", zorder=0)
    for k in clamp_ev:
        ax1.axvline(0.5 * (th[k] + th[k + 1]), color=TEAL, lw=0.8, ls=(0, (2.0, 1.5)), zorder=0)
    ax1.plot(th[valid], yd[valid], ".", ms=2.6, color=INK, zorder=2)
    ax1.set_xlabel(r"$\theta_B$ (deg)")
    ax1.set_ylabel(r"$J_3'$ minus linear fit" + "\n" + r"($10^{-3}\,R_t$/deg)")
    ax1.text(0.015, 0.97, "(a)", transform=ax1.transAxes, va="top", ha="left", fontsize=9,
             bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    h1 = [plt.Line2D([], [], ls="none", marker=".", ms=5, color=INK, label="AD derivative (E12 sweep)"),
          plt.Line2D([], [], color=ACCENT, lw=0.9, label="row-count change"),
          plt.Line2D([], [], color=TEAL, lw=0.9, ls=(0, (2.0, 1.5)), label=r"TH3$\geq$0 clamp switch")]
    ax1.legend(handles=h1, loc="lower left", bbox_to_anchor=(-0.02, 1.02), ncol=2, frameon=False,
               handlelength=1.6, columnspacing=1.0, borderaxespad=0.0)

    floors = {}
    for q, nm in enumerate(NAMES):
        fl = []
        for tp in res["taylor"]:
            o = tp["outputs"][nm]
            E = np.array(o["E"])
            m = E > 0
            ax2.loglog(H_DEG[m], E[m], "-", marker="o", ms=3.0, lw=0.8, color=COLORS[q], mfc=COLORS[q],
                       mec="white", mew=0.4, zorder=3)
            fl.append(o["noise_floor"])
        floors[nm] = NOISE_FACTOR * max(fl)
        ax2.axhline(floors[nm], color=COLORS[q], lw=0.8, ls=":", zorder=1)
    tb = rb["taylor"][E12B_POINT]
    hb = np.array(tb["h_deg"])
    j1 = tb["outputs"]["J1_mdot_BD"]
    Ef, Eb = np.array(j1["forward"]["E"]), np.array(j1["backward"]["E"])
    ax2.loglog(hb[Ef > 0], Ef[Ef > 0], "--", marker="D", ms=3.4, lw=0.8, color=INK, mfc=INK, mec=INK, zorder=4)
    ax2.loglog(hb[Eb > 0], Eb[Eb > 0], "-.", marker="D", ms=3.4, lw=0.8, color=INK, mfc="white", mec=INK, zorder=4)
    for val, ls in ((NOISE_FACTOR * j1["forward"]["noise_floor"], "--"), (NOISE_FACTOR * j1["backward"]["noise_floor"], "-.")):
        ax2.plot([2e-5, 6e-4], [val, val], ls=ls, color=MUTED, lw=1.5, zorder=2)
    href = np.array([1e-4, 1e-1])
    anchor = res["taylor"][0]["outputs"]["J3_x_D_axis"]["E"][2]
    ax2.loglog(href, anchor * (href / 1e-3) ** 2 * 4.0, color=MUTED, lw=0.9, ls=(0, (5, 2)))
    ax2.set_xlabel(r"step $h$ (deg)")
    ax2.set_ylabel(r"$E(h)=|J(t\pm h)-J(t)\mp hJ'(t)|$")
    ax2.set_ylim(1e-18, 1e-1)
    ax2.text(0.015, 0.97, "(b)", transform=ax2.transAxes, va="top", ha="left", fontsize=9,
             bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    h2 = [plt.Line2D([], [], color=COLORS[q], lw=0.9, marker="o", ms=3, label=LABELS[q] + " E12") for q in range(3)]
    h2 += [plt.Line2D([], [], color=MUTED, lw=0.9, ls=(0, (5, 2)), label="slope 2"),
           plt.Line2D([], [], color=MUTED, lw=0.9, ls=":", label=r"100$\times$ floor"),
           (plt.Line2D([], [], color=INK, lw=0.8, ls="--", marker="D", ms=3),
            plt.Line2D([], [], color=INK, lw=0.8, ls="-.", marker="D", ms=3, mfc="white"))]
    labels2 = [hh.get_label() for hh in h2[:5]] + [r"$J_1$ E12b fwd/bwd"]
    from matplotlib.legend_handler import HandlerTuple
    ax2.legend(handles=h2, labels=labels2, loc="lower left", bbox_to_anchor=(-0.02, 1.02), ncol=3, frameon=False,
               handlelength=1.6, columnspacing=0.8, borderaxespad=0.0, fontsize=8,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.4)})
    base = os.path.join(OUT, "e12_print")
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=200)
    plt.close(fig)

    script = os.path.abspath(__file__)
    pf = dict(
        script=os.path.relpath(script, ROOT), script_sha256=sha(script),
        command=".venv-jax/bin/python " + os.path.relpath(script, ROOT), git_head=git_head(),
        inputs={os.path.relpath(npz_path, ROOT): sha(npz_path),
                os.path.relpath(res_path, ROOT) + " (before this amendment)": res_sha_before,
                os.path.relpath(b_path, ROOT): sha(b_path)},
        figure=dict(pdf=os.path.relpath(base + ".pdf", ROOT), pdf_sha256=sha(base + ".pdf"),
                    png=os.path.relpath(base + ".png", ROOT), png_sha256=sha(base + ".png"), size_in=[6.8, 2.6]),
        panels=[
            "(a) E12 sweep: AD derivative dJ3/dtheta_B minus its linear fit vs theta_B (27.0-28.5 deg, 301 points); "
            "12 row-count changes (solid accent) and 12 TH3>=0 clamp switches (dashed teal), event classes from "
            "e12b/result.json event_free_intervals and e12_sweep.npz pattern_change; J2's derivative shows no visible "
            "events at this scale and is not plotted",
            "(b) Taylor remainders vs h: E12 pre-registered forward E(h) for J1, J2, J3 at theta_B = 27.23, 27.73, 28.23 "
            "deg (solid, filled circles) with each output's pre-registered 100x noise floor (dotted, max over the three "
            "points); POST-HOC E12b J1 at theta_B = %.4f deg forward (dashed, filled diamonds) and backward "
            "(dash-dot, open diamonds) with their own 100x floors (short grey segments); slope-2 reference" % tb["theta_B_deg"]],
        caption_facts=dict(
            configuration=("Print re-plot of E12 from saved data (no new JAX runs): JAX throat kernel, Nozzle-B, n 24, "
                           "dtheta 0.25 deg. Panel (b) mixes pre-registered E12 data with POST-HOC E12b J1 data, and "
                           "the line styles keep them distinct."),
            evidence_class="verification",
            not_established="E12 status unchanged (FAIL); E12b is post hoc and cannot change it; no derivative of the full net, start line or BVP."),
        pre_amendment_result_sha256=res_sha_before,
        wall_time_s=time.time() - t0, status="unchanged (figures only)",
        note=("result.json was amended only by adding this print_figures key; the e12 diagnostic and e12b inputs "
              "hashed the pre-amendment file (sha256 given above)."),
    )
    res["print_figures"] = pf
    with open(res_path, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(dict(figure=pf["figure"], result_sha256_after=sha(res_path)), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
