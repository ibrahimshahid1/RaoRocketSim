"""Print versions (6.8 in wide, >= 8 pt) of the B2 E05 and E10 figures, re-plotted from saved
result.json data only (no MOC runs). The E10 bell contour is regenerated with
lrekit.components.nozzle.geometry.bell_nozzle_contour (geometry only). Numbers and statuses are unchanged;
each result.json gains a print_figures block.

    .venv-jax/bin/python tests/experiments/nozzle/moc_ext_print_figures_b2.py
"""
from __future__ import annotations

from lrekit.paths import repository_root as _repository_root

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = _repository_root()
sys.path.insert(0, str(ROOT))
B = ROOT / "builds/moc_extensions_plan_2026-11-21/opus_b"

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

INK, TEAL, MUTED, ACCENT = "#19333C", "#176D76", "#51616B", "#C26A2E"
COMMAND = ".venv-jax/bin/python tests/experiments/nozzle/moc_ext_print_figures_b2.py"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8, "axes.edgecolor": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "pdf.fonttype": 42, "axes.titlesize": 8,
})


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(fig, out_dir: Path, stem: str) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=200)
        paths.append(str(p.relative_to(ROOT)))
    plt.close(fig)
    return paths


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ------------------------------------------------------------------ E05
CASES = (("alpha15_gamma1.23", "cone 15\u00b0, $\\gamma$ = 1.23", INK),
         ("alpha15_gamma1.4", "cone 15\u00b0, $\\gamma$ = 1.4", TEAL),
         ("alpha25_gamma1.23", "cone 25\u00b0, $\\gamma$ = 1.23", MUTED),
         ("alpha25_gamma1.4", "cone 25\u00b0, $\\gamma$ = 1.4", ACCENT))


def e05_print():
    res_path = B / "e05/result.json"
    d = json.loads(res_path.read_text())
    spec = d["configuration"]["spec_levels"]
    crit = spec[-3:]
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.5), layout="constrained")
    hmin = hmax = None
    ymin = {0: np.inf, 1: np.inf}
    for key, lab, col in CASES:
        lv = [r for r in d["metrics"][key]["levels"] if r["n"] in spec and r.get("complete")]
        h = np.array([r["h_rel"] for r in lv])
        hmin = h.min() if hmin is None else min(hmin, h.min())
        hmax = h.max() if hmax is None else max(hmax, h.max())
        for ax, q in zip(axs, ("max_dM", "max_dtheta_rad")):
            for scope, ls, mfc in (("errors_all_nodes", "-", col), ("errors_excluding_exit_lip_row", "--", "white")):
                y = np.array([max(r[scope][k][q] for k in ("interior", "axis", "wall") if r[scope].get(k))
                              for r in lv])
                if q == "max_dtheta_rad":
                    y = np.degrees(y)
                i_ax = 0 if q == "max_dM" else 1
                ymin[i_ax] = min(ymin[i_ax], float(y.min()))
                ax.loglog(h, y, ls=ls, marker="o", ms=3.2, lw=1.0, color=col, mfc=mfc, mec=col)
    hc = [1.0 / (n - 1) for n in crit]
    for i_ax, ax in enumerate(axs):
        ax.axvspan(min(hc) / 1.25, max(hc) * 1.25, color=MUTED, alpha=0.13, lw=0, zorder=0)
        y0 = 0.3 * ymin[i_ax]
        hh = np.array([hmin, hmax])
        ax.loglog(hh, y0 * (hh / hmin) ** 2.0, ":", color=MUTED, lw=1.0)
        ax.loglog(hh, y0 * (hh / hmin) ** 1.5, "-.", color=MUTED, lw=1.0)
        ax.set_xlabel("start-line spacing $h = 1/(n-1)$ (-)")
        clean(ax)
    axs[0].set_ylabel("max |M error| over nodes (-)")
    axs[1].set_ylabel("max |flow-angle error| (deg)")
    handles = [Line2D([], [], color=c, lw=1.2, marker="o", ms=3.2, label=l) for _, l, c in CASES]
    handles += [Line2D([], [], color=INK, lw=1.0, ls="-", marker="o", ms=3.2, label="all nodes (pre-registered)"),
                Line2D([], [], color=INK, lw=1.0, ls="--", marker="o", ms=3.2, mfc="white",
                       label="exit-lip row excluded (POST-HOC)"),
                Line2D([], [], color=MUTED, lw=1.0, ls=":", label="order 2"),
                Line2D([], [], color=MUTED, lw=1.0, ls="-.", label="order 1.5"),
                Patch(color=MUTED, alpha=0.13, label=f"order-criterion levels (n {', '.join(map(str, crit))})")]
    order = list(range(9))
    fig.legend(handles=[handles[i] for i in order], loc="outside upper center", ncol=3, frameon=False,
               handlelength=2.2, columnspacing=1.2)
    paths = save(fig, B / "e05/print", "e05_print_convergence_max")
    return res_path, d, paths, crit


# ------------------------------------------------------------------ E10
def e10_print():
    from lrekit.components.nozzle.geometry import bell_nozzle_contour
    res_path = B / "e10/result.json"
    d = json.loads(res_path.read_text())
    tab = d["metrics"]["reproduction"]["table"]
    Rt = 0.02
    c = bell_nozzle_contour(Rt=Rt, epsilon=10.0, length_pct=80.0, gamma=1.2, theta_n_deg=None, theta_e_deg=None)
    fig = plt.figure(figsize=(6.8, 2.6), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0])
    ax = fig.add_subplot(gs[:, 0])
    axx = fig.add_subplot(gs[0, 1])
    axr = fig.add_subplot(gs[1, 1], sharex=axx)
    x = np.asarray(c["x"]) / Rt
    y = np.asarray(c["y"]) / Rt
    m = x >= 0
    ax.plot(x[m], y[m], color=INK, lw=1.1)
    ax.axhline(0.0, color=MUTED, lw=0.6)
    mk = {24: "o", 48: "s", 96: "^", 144: "D"}
    ref = [t for t in tab if t["on_invalid"] == "refine"]
    stp = [t for t in tab if t["on_invalid"] == "stop"]
    for t in ref:
        ax.plot(t["fold"][0] / Rt, t["fold"][1] / Rt, mk[t["n"]], ms=4.5, color=TEAL, mec=INK, mew=0.6)
    N = c["N"]
    ax.plot(N[0] / Rt, N[1] / Rt, "P", ms=6, color=ACCENT, mec=INK, mew=0.4)
    ax.set_xlim(-0.2, x.max() * 1.03)
    ax.set_ylim(-0.1, y.max() * 1.08)
    ax.set_xlabel("$x/R_t$ (-)")
    ax.set_ylabel("$r/R_t$ (-)")
    clean(ax)
    ins = ax.inset_axes([0.55, 0.24, 0.43, 0.34])
    for t in ref:
        ins.plot(t["fold"][0] / Rt, t["fold"][1] / Rt, mk[t["n"]], ms=4.5, color=TEAL, mec=INK, mew=0.6)
    for t in stp:
        ins.plot(t["fold"][0] / Rt, t["fold"][1] / Rt, mk[t["n"]], ms=4.5, mfc="white", mec=MUTED, mew=0.8)
    ins.set_xlim(2.48, 2.86)
    ins.set_ylim(0.934, 0.954)
    ins.set_xticks([2.5, 2.8])
    ins.set_yticks([0.94, 0.95])
    ins.tick_params(labelsize=8, length=2, pad=1)
    for s in ins.spines.values():
        s.set_color(MUTED)
    ax.indicate_inset_zoom(ins, edgecolor=MUTED, lw=0.6)
    for a_, key in ((axx, 0), (axr, 1)):
        for rows, ls, col, mfc in ((ref, "-", TEAL, TEAL), (stp, "--", MUTED, "white")):
            h = np.array([t["dtheta_deg"] for t in rows])
            v = np.array([t["fold"][key] / Rt for t in rows])
            a_.plot(h, v, ls=ls, color=col, lw=1.0)
            for hh, vv, t in zip(h, v, rows):
                a_.plot(hh, vv, mk[t["n"]], ms=4.0, color=col, mfc=mfc, mec=col if mfc == "white" else INK,
                        mew=0.7)
        a_.set_xscale("log")
        a_.set_xticks([t["dtheta_deg"] for t in ref])
        a_.set_xticklabels([f"{t['dtheta_deg']:g}" for t in ref])
        a_.minorticks_off()
        a_.invert_xaxis() if key == 0 else None
        clean(a_)
    axx.set_ylabel("fold $x/R_t$")
    axr.set_ylabel("fold $r/R_t$")
    axr.set_xlabel(r"wall-turning limit $\Delta\theta_B$ (deg)")
    plt.setp(axx.get_xticklabels(), visible=False)
    handles = [Line2D([], [], color=INK, lw=1.1, label="bell contour (Bezier, 25.74/10.49 deg)"),
               Line2D([], [], ls="", marker="P", ms=6, color=ACCENT, mec=INK, mew=0.4, label="N (bell start)")]
    handles += [Line2D([], [], ls="", marker=mk[n], ms=4.5, color=TEAL, mec=INK, mew=0.6, label=f"fold n {n}")
                for n in (24, 48, 96, 144)]
    handles += [Line2D([], [], color=TEAL, lw=1.0, ls="-", label="on_invalid 'refine' (t01, primary)"),
                Line2D([], [], color=MUTED, lw=1.0, ls="--", marker="o", ms=4, mfc="white",
                       label="on_invalid 'stop' (sensitivity)")]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False, handlelength=2.0,
               columnspacing=1.0)
    paths = save(fig, B / "e10/print", "e10_print_fold_contour_spacing")
    return res_path, d, paths


def record(res_path, d, paths, panels, notes):
    script = Path(__file__).resolve()
    import lrekit  # noqa: F401
    rao = {}
    for name, mod in sorted(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if name.startswith("lrekit") and f and f.endswith(".py"):
            rao[str(Path(f).resolve().relative_to(ROOT))] = sha256(Path(f))
    d["print_figures"] = {
        "script": str(script.relative_to(ROOT)), "script_sha256": sha256(script), "command": COMMAND,
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                   text=True).stdout.strip(),
        "source_data": str(res_path.relative_to(ROOT)), "raosim_imported_sha256": rao,
        "files": {p: sha256(ROOT / p) for p in paths}, "size_in": None, "min_font_pt": 8,
        "panels": panels, "notes": notes,
        "numbers_or_statuses_changed": False,
    }
    res_path.write_text(json.dumps(d, indent=1, default=float), encoding="utf-8")


def main():
    rp, d, paths, crit = e05_print()
    d["print_figures"] = None
    record(rp, d, paths, [
        "(a) left: max |M error| over all nodes (max of interior, axis and wall classes) vs h for the four "
        "cone/gamma cases at the spec levels n 13-193; solid = all nodes (pre-registered FAIL basis), dashed open = "
        "exit-lip closure row excluded (POST-HOC); order-1.5 and order-2 reference slopes; grey band = the "
        f"pre-registered order-criterion levels n {', '.join(map(str, crit))}",
        "(b) right: same for max |flow-angle error| (deg)",
    ], ["Supplementary level n = 385 (outside acceptance) is omitted from the print figure.",
        "Re-plotted from e05/result.json only; no new runs."])
    d0 = json.loads(rp.read_text())
    d0["print_figures"]["size_in"] = [6.8, 2.5]
    rp.write_text(json.dumps(d0, indent=1, default=float), encoding="utf-8")
    rp, d, paths = e10_print()
    record(rp, d, paths, [
        "(a) left: bell contour (x/Rt, r/Rt) with N and the four on_invalid 'refine' fold points; inset zoom of "
        "the fold cluster with the 'stop' sensitivity folds (open markers); legend outside the axes",
        "(b) right top: fold x/Rt vs wall-turning limit (n 24/48/96/144), 'refine' solid, 'stop' dashed",
        "(c) right bottom: fold r/Rt vs wall-turning limit, same series",
    ], ["Fold data from e10/result.json (metrics.reproduction.table); the bell contour is regenerated with "
        "lrekit.components.nozzle.geometry.bell_nozzle_contour (geometry only, no MOC run).",
        "The 'last minus second-last stored C-' gap curve belongs to e10a's right panel; its curve data were "
        "not saved, so it is not re-plotted here (no new runs). All legends sit outside the axes."])
    d0 = json.loads(rp.read_text())
    d0["print_figures"]["size_in"] = [6.8, 2.6]
    rp.write_text(json.dumps(d0, indent=1, default=float), encoding="utf-8")
    print("ok")


if __name__ == "__main__":
    main()
