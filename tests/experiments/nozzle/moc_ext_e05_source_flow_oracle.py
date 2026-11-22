"""E05: exact radial source-flow oracle for the host bounded forward net.

Inviscid radial flow from an apex at the origin inside a straight cone of
half-angle alpha: theta = atan(r/x), A/A* = (R/R*)^2 with R = sqrt(x^2 + r^2).
A/A* = (R/R*)^2 is a derivation composed from sivells1978 (PDF 16 / printed 13,
Eq. 17, source-flow sonic area 2 pi r1^2 (1 - cos eta); PDF 8 radial flow from a
source) and anderson Eq. 5.20 (PDF 220 / printed 204), not a page quote.
Opt-in diagnostic; changes nothing in raosim/.

    .venv-jax/bin/python tests/experiments/nozzle/moc_ext_e05_source_flow_oracle.py [--levels 13,25,49,97,193]
"""
from __future__ import annotations

from lrekit.paths import repository_root as _repository_root

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = _repository_root()
sys.path.insert(0, str(ROOT))
OUT = ROOT / "builds/moc_extensions_plan_2026-11-21/opus_b/e05"

import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from lrekit.components.nozzle.solvers.bounded import crop_to_exit_plane, exit_plane_mass_flow, march_bounded_net  # noqa: E402
from lrekit.components.nozzle.solvers.nasa import _char_point_from_values  # noqa: E402

INK, TEAL, MUTED, ACCENT = "#19333C", "#176D76", "#51616B", "#C26A2E"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raosim_hashes() -> dict:
    out = {}
    for name, mod in sorted(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if name.startswith("lrekit") and f and f.endswith(".py"):
            out[str(Path(f).resolve().relative_to(ROOT))] = sha256(Path(f))
    return out


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


# ------------------------------------------------------------- exact field
def area_ratio(M: float, g: float) -> float:
    return (1.0 / M) * ((2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * M * M)) ** (
        (g + 1.0) / (2.0 * (g - 1.0)))


def mach_of_R(R: float, g: float) -> float:
    target = R * R
    if target <= 1.0:
        return 1.0
    hi = 2.0
    while area_ratio(hi, g) < target:
        hi *= 2.0
    return brentq(lambda m: area_ratio(m, g) - target, 1.0, hi, xtol=1e-15, rtol=8.9e-16,
                  maxiter=500)


def exact_state(x: float, r: float, g: float) -> tuple[float, float]:
    return mach_of_R(math.hypot(x, r), g), math.atan2(r, x)


def exact_mdot(alpha: float, g: float) -> float:
    rho_s = (2.0 / (g + 1.0)) ** (1.0 / (g - 1.0))
    V_s = math.sqrt(g * 2.0 / (g + 1.0))
    return rho_s * V_s * 2.0 * math.pi * (1.0 - math.cos(alpha))


class ConeWall:
    def __init__(self, alpha: float, x_start: float, x_end: float):
        self.alpha, self.t = alpha, math.tan(alpha)
        self.x_start, self.x_end = float(x_start), float(x_end)

    def r(self, x):
        return float(x) * self.t

    def dr_dx(self, x):
        return self.t

    def theta(self, x):
        return self.alpha


def cminus_dxdr(x: float, r: float, g: float) -> float:
    M, th = exact_state(x, r, g)
    return 1.0 / math.tan(th - math.asin(1.0 / M))


def exact_cminus_to_axis(x0: float, r0: float, g: float, r_nodes: np.ndarray,
                         sub: int) -> np.ndarray:
    """RK4 in r along the exact C- (dx/dr = cot(theta - mu)) through the r_nodes (descending)."""
    xs = [x0]
    x = x0
    for ra, rb in zip(r_nodes[:-1], r_nodes[1:]):
        h = (rb - ra) / sub
        r = ra
        for _ in range(sub):
            k1 = cminus_dxdr(x, r, g)
            k2 = cminus_dxdr(x + 0.5 * h * k1, r + 0.5 * h, g)
            k3 = cminus_dxdr(x + 0.5 * h * k2, r + 0.5 * h, g)
            k4 = cminus_dxdr(x + h * k3, r + h, g)
            x += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            r += h
        xs.append(x)
    return np.array(xs)


def exact_cminus_axis_foot(x0: float, r0: float, g: float, dx: float = 2e-4) -> float:
    """Axis foot of the exact C- from (x0, r0), RK4 in x (dr/dx = tan(theta - mu); r need not be monotone)."""
    def f(x, r):
        M, th = exact_state(x, max(r, 0.0), g)
        return math.tan(th - math.asin(1.0 / M))
    x, r = x0, r0
    while True:
        k1 = f(x, r)
        k2 = f(x + 0.5 * dx, r + 0.5 * dx * k1)
        k3 = f(x + 0.5 * dx, r + 0.5 * dx * k2)
        k4 = f(x + dx, r + dx * k3)
        rn = r + dx * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        if rn <= 0.0:
            return x + dx * r / (r - rn)
        x, r = x + dx, rn


def build_case(alpha_deg: float, g: float, n: int, M0: float = 1.5, Mexit: float = 3.0,
               total_sub: int = 24000):
    alpha = math.radians(alpha_deg)
    R0 = math.sqrt(area_ratio(M0, g))
    Rex = math.sqrt(area_ratio(Mexit, g))
    x0, r0 = R0 * math.cos(alpha), R0 * math.sin(alpha)
    r_nodes = np.linspace(r0, 0.0, n)
    sub = max(1, math.ceil(total_sub / (n - 1)))
    xs = exact_cminus_to_axis(x0, r0, g, r_nodes, sub)
    pts = []
    for x, r in zip(xs, r_nodes):
        M, th = exact_state(x, r, g)
        if r == 0.0:
            th = 0.0
        pts.append(_char_point_from_values(x, r, th, M, g))
    start_axis_first = list(reversed(pts))
    wall = ConeWall(alpha, x0, Rex * math.cos(alpha))
    return start_axis_first, wall, {"alpha_deg": alpha_deg, "gamma": g, "n": n, "R0": R0,
                                    "R_exit": Rex, "x0": x0, "r0": r0, "x_exit": wall.x_end,
                                    "rk4_substeps_per_interval": sub}


def lip_rows(net) -> set:
    return {j for j in range(1, net.n_rows)
            if net.parents[j] and net.parents[j][0].get("foot") == "exit_lip"}


def node_errors(net, g):
    lips = lip_rows(net)
    out = []
    for j, row in enumerate(net.rows):
        if j == 0:
            continue
        for k, p in enumerate(row):
            kind = "wall" if k == 0 else ("axis" if abs(p.r) <= 1e-12 else "interior")
            Me, the = exact_state(p.x, p.r, g)
            if kind == "axis":
                the = 0.0
            out.append((j, k, kind, j in lips, p.x, p.r, p.M - Me, p.theta - the))
    return out


def norms(errs, kind, exclude_lip):
    sel = [e for e in errs if e[2] == kind and not (exclude_lip and e[3])]
    if not sel:
        return None
    dM = np.array([e[6] for e in sel])
    dth = np.array([e[7] for e in sel])
    return {"n": len(sel), "max_dM": float(np.max(np.abs(dM))), "rms_dM": float(np.sqrt(np.mean(dM ** 2))),
            "max_dtheta_rad": float(np.max(np.abs(dth))),
            "rms_dtheta_rad": float(np.sqrt(np.mean(dth ** 2)))}


def interp_axis_M(net, x_probe):
    ax = sorted((row[-1].x, row[-1].M) for row in net.rows[1:] if abs(row[-1].r) <= 1e-12)
    xa, Ma = np.array([a for a, _ in ax]), np.array([b for _, b in ax])
    if not (xa[0] <= x_probe <= xa[-1]):
        return None
    return float(np.interp(x_probe, xa, Ma))


def interp_wall_M(net, x_probe):
    w = sorted((row[0].x, row[0].M) for row in net.rows[1:])
    xa, Ma = np.array([a for a, _ in w]), np.array([b for _, b in w])
    if not (xa[0] <= x_probe <= xa[-1]):
        return None
    return float(np.interp(x_probe, xa, Ma))


def run_level(alpha_deg, g, n):
    start, wall, meta = build_case(alpha_deg, g, n)
    t0 = time.time()
    err_msg = None
    try:
        net = march_bounded_net(start, wall, g, arc=None, dtheta_limit=math.radians(0.5),
                                mass_tol=0.02, on_invalid="stop")
    except Exception as exc:
        net, err_msg = None, f"{type(exc).__name__}: {exc}"
    return start, wall, meta, net, time.time() - t0, err_msg


def celik_gci(phi1, phi2, phi3, r=2.0, Fs=1.25):
    """Fine (1), medium (2), coarse (3) at constant ratio r; celik2008 (equation pages pending LR4)."""
    if None in (phi1, phi2, phi3):
        return {"p": None, "missing_probe": True, "phi": [phi1, phi2, phi3]}
    e21, e32 = phi2 - phi1, phi3 - phi2
    if e21 == 0 or e32 == 0 or e32 / e21 <= 0:
        return {"p": None, "oscillatory_or_zero": True, "phi": [phi1, phi2, phi3]}
    p = abs(math.log(abs(e32 / e21))) / math.log(r)
    phi_ext = (r ** p * phi1 - phi2) / (r ** p - 1.0)
    ea21 = abs((phi1 - phi2) / phi1)
    return {"p": p, "phi_ext": phi_ext, "e_a21": ea21, "GCI_fine": Fs * ea21 / (r ** p - 1.0),
            "phi": [phi1, phi2, phi3]}


def orders(vals):
    return [math.log2(a / b) if (a is not None and b is not None and a > 0 and b > 0) else None
            for a, b in zip(vals[:-1], vals[1:])]


def run_config(alpha_deg, g, levels):
    lv, nets = [], {}
    mdot_ex = exact_mdot(math.radians(alpha_deg), g)
    alpha = math.radians(alpha_deg)
    R0, Rex = math.sqrt(area_ratio(1.5, g)), math.sqrt(area_ratio(3.0, g))
    xa0 = exact_cminus_axis_foot(R0 * math.cos(alpha), R0 * math.sin(alpha), g)
    xa1 = exact_cminus_axis_foot(Rex * math.cos(alpha), Rex * math.sin(alpha), g)
    x_ax = 0.5 * (xa0 + xa1)
    x_w = 0.5 * (R0 + Rex) * math.cos(alpha)
    for n in levels:
        start, wall, meta, net, wt, err = run_level(alpha_deg, g, n)
        row = {"n": n, "h_rel": 1.0 / (n - 1), "h_r_over_Rstar": meta["r0"] / (n - 1),
               "wall_time_s": wt, "meta": meta}
        if net is None:
            row.update({"complete": False, "error": err})
            lv.append(row)
            continue
        errs = node_errors(net, g)
        lips = sorted(lip_rows(net))
        mdot_exit = exit_plane_mass_flow(crop_to_exit_plane(net), g)
        lip_detail = []
        for j in lips:
            rw = net.rows[j]
            Me, _ = exact_state(rw[-1].x, 0.0, g)
            lip_detail.append({"row": j, "axis_x": rw[-1].x, "axis_dM": rw[-1].M - Me,
                               "last_interior_r": rw[-2].r if len(rw) > 2 else None,
                               "prev_row_last_interior_r": net.rows[j - 1][-2].r})
        row.update({
            "complete": bool(net.complete), "stop_reason": net.stop_reason, "n_rows": net.n_rows,
            "n_special_rows": int(sum(net.special)), "exit_lip_rows": lips, "exit_lip_detail": lip_detail,
            "n_refined": net.n_refined, "containment_violations": net.containment_violations,
            "truncated_rows": net.truncated_rows, "fold_location": net.fold_location,
            "errors_all_nodes": {k: norms(errs, k, False) for k in ("interior", "axis", "wall")},
            "errors_excluding_exit_lip_row": {k: norms(errs, k, True) for k in ("interior", "axis", "wall")},
            "row_mdot_rel_err_vs_exact": [float((m - mdot_ex) / mdot_ex) for m in net.massflow],
            "row_mdot_rel_err_vs_row0": [float(x) for x in net.mdot_rel_err],
            "exit_plane_mdot_rel_err_vs_exact": float((mdot_exit - mdot_ex) / mdot_ex),
            "exit_plane_mdot": float(mdot_exit),
            "probe_axis_M": interp_axis_M(net, x_ax), "probe_wall_M": interp_wall_M(net, x_w),
        })
        lv.append(row)
        nets[n] = (net, errs)
    return lv, nets, mdot_ex, {"x_axis_probe": x_ax, "x_axis_probe_exact_M": mach_of_R(x_ax, g),
                               "x_wall_probe": x_w,
                               "x_wall_probe_exact_M": mach_of_R(x_w / math.cos(alpha), g),
                               "rule": "axis probe midway between the exact axis feet of the start C- and the "
                                       "exit-lip C-; wall probe midway between R0 and R_exit"}


def summarize(lv, spec_levels):
    ok = [r for r in lv if r.get("complete") and r["n"] in spec_levels]
    s = {"all_spec_levels_complete": len(ok) == len(spec_levels)}
    for scope in ("errors_all_nodes", "errors_excluding_exit_lip_row"):
        d = {}
        for kind in ("interior", "axis", "wall"):
            for q in ("max_dM", "rms_dM", "max_dtheta_rad", "rms_dtheta_rad"):
                vals = [r[scope][kind][q] if r[scope].get(kind) else None for r in ok]
                d[f"{kind}_{q}"] = vals
                d[f"{kind}_{q}_order"] = orders(vals)
                d[f"{kind}_{q}_monotone"] = None not in vals and all(a > b for a, b in zip(vals[:-1], vals[1:]))
        s[scope] = d
    if len(ok) >= 3:
        f1, f2, f3 = ok[-1], ok[-2], ok[-3]
        s["gci_three_finest"] = {
            "levels": [f3["n"], f2["n"], f1["n"]],
            "axis_M_probe": celik_gci(f1["probe_axis_M"], f2["probe_axis_M"], f3["probe_axis_M"]),
            "wall_M_probe": celik_gci(f1["probe_wall_M"], f2["probe_wall_M"], f3["probe_wall_M"]),
            "exit_plane_mdot": celik_gci(f1["exit_plane_mdot"], f2["exit_plane_mdot"], f3["exit_plane_mdot"]),
        }
    return s


def acceptance(summary):
    out = {}
    for scope in ("errors_all_nodes", "errors_excluding_exit_lip_row"):
        d = summary[scope]
        mono = all(d[f"{k}_{q}_monotone"] for k in ("interior", "axis") for q in ("max_dM", "rms_dM"))
        mono = mono and all(d[f"interior_{q}_monotone"] for q in ("max_dtheta_rad", "rms_dtheta_rad"))
        p_int = [o for q in ("max_dM", "rms_dM", "max_dtheta_rad", "rms_dtheta_rad")
                 for o in (d[f"interior_{q}_order"][-2:] if d[f"interior_{q}_order"] else [None])]
        p_int_min = min(o for o in p_int if o is not None) if any(o is not None for o in p_int) else None
        p_ax = [o for q in ("max_dM", "rms_dM") for o in (d[f"axis_{q}_order"][-2:] or [None])]
        p_ax_min = min(o for o in p_ax if o is not None) if any(o is not None for o in p_ax) else None
        p_int_M = [o for q in ("max_dM", "rms_dM") for o in (d[f"interior_{q}_order"][-2:] or [None])]
        p_int_M_min = min(o for o in p_int_M if o is not None) if any(o is not None for o in p_int_M) else None
        out[scope] = {
            "monotone_decrease_M_and_theta": mono,
            "min_interior_order_finest_two_pairs": p_int_min,
            "min_axis_M_order_finest_two_pairs": p_ax_min,
            "interior_order_ge_1.5": p_int_min is not None and p_int_min >= 1.5,
            "axis_vs_interior_M_order_gap": (None if (p_ax_min is None or p_int_M_min is None)
                                             else p_int_M_min - p_ax_min),
            "MOC14_active_by_criterion": (p_ax_min is not None and p_int_M_min is not None
                                          and p_int_M_min - p_ax_min > 0.5),
        }
    return out


# ------------------------------------------------------------- figures
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8.5, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "pdf.fonttype": 42,
    })
    return plt


def _save(plt, fig, stem):
    paths = []
    for ext in ("pdf", "png"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(str(p.relative_to(ROOT)))
    plt.close(fig)
    return paths


def fig_convergence(plt, results, qty, stem, spec_levels):
    keys = list(results)
    fig, axs = plt.subplots(2, 2, figsize=(6.5, 4.6), sharex=True, constrained_layout=True)
    for ax, key in zip(axs.flat, keys):
        lv = [r for r in results[key]["levels"] if r.get("complete")]
        h = np.array([r["h_rel"] for r in lv])
        def get(scope, kind, q):
            return np.array([r[scope][kind][q] for r in lv])
        if qty == "M":
            series = [("errors_all_nodes", "interior", "max_dM", INK, "o", "-", "interior max, all rows"),
                      ("errors_excluding_exit_lip_row", "interior", "max_dM", INK, "o", "--",
                       "interior max, exit-lip row excluded"),
                      ("errors_all_nodes", "interior", "rms_dM", TEAL, "^", "-", "interior RMS"),
                      ("errors_all_nodes", "axis", "max_dM", ACCENT, "s", "-", "axis max, all rows"),
                      ("errors_excluding_exit_lip_row", "axis", "max_dM", ACCENT, "s", "--",
                       "axis max, exit-lip row excluded")]
        else:
            series = [("errors_all_nodes", "interior", "max_dtheta_rad", INK, "o", "-", "interior max, all rows"),
                      ("errors_excluding_exit_lip_row", "interior", "max_dtheta_rad", INK, "o", "--",
                       "interior max, exit-lip row excluded"),
                      ("errors_all_nodes", "interior", "rms_dtheta_rad", TEAL, "^", "-", "interior RMS")]
        for scope, kind, q, c, m, ls, lab in series:
            y = get(scope, kind, q)
            if qty != "M":
                y = np.degrees(y)
            ax.loglog(h, y, ls=ls, marker=m, ms=3.5, color=c, lw=1.0, mfc="white" if ls == "--" else c,
                      label=lab)
        spec_mask = np.array([r["n"] in spec_levels for r in lv])
        if (~spec_mask).any():
            ax.axvspan(h[~spec_mask].min() * 0.8, h[~spec_mask].max() * 1.25, color=MUTED, alpha=0.12, lw=0)
        y0 = get("errors_all_nodes", "interior", "rms_dM" if qty == "M" else "rms_dtheta_rad")
        if qty != "M":
            y0 = np.degrees(y0)
        ax.loglog(h, y0[-1] * 0.5 * (h / h[-1]) ** 2, color=MUTED, lw=0.8, ls=":", label="slope 2")
        a, g = key.replace("alpha", "").replace("gamma", "").split("_")
        ax.text(0.04, 0.94, f"cone {a} deg, gamma {g}", transform=ax.transAxes, va="top", fontsize=7.5,
                color=INK)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    for ax in axs[1]:
        ax.set_xlabel("start-line spacing $h$ = 1/(n-1) (-)")
    for ax in axs[:, 0]:
        ax.set_ylabel("|M error| (-)" if qty == "M" else "|flow-angle error| (deg)")
    h_, l_ = axs[0, 0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    return _save(plt, fig, stem)


def fig_map(plt, nets, keys, stem):
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.6), constrained_layout=True)
    import matplotlib.colors as mcolors
    norm = mcolors.LogNorm(vmin=1e-7, vmax=3e-3)
    sc = None
    for ax, key in zip(axs, keys):
        net, errs = nets[key]
        x = np.array([e[4] for e in errs])
        r = np.array([e[5] for e in errs])
        dm = np.clip(np.abs(np.array([e[6] for e in errs])), 1e-12, None)
        lip = np.array([e[3] for e in errs])
        sc = ax.scatter(x[~lip], r[~lip], c=dm[~lip], s=1.2, norm=norm, cmap="cividis", lw=0)
        ax.scatter(x[lip], r[lip], c=dm[lip], s=2.5, norm=norm, cmap="cividis", marker="s",
                   edgecolors=ACCENT, linewidths=0.3)
        xw = np.array([row[0].x for row in net.rows])
        ax.plot(xw, xw * math.tan(net.rows[1][0].theta), color=INK, lw=0.8)
        a, g = key.replace("alpha", "").replace("gamma", "").split("_")
        ax.set_title("")
        ax.text(0.98, 0.94, f"cone {a} deg, gamma {g}", transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color=INK)
        ax.set_xlabel("$x/R^*$ (-)")
        ax.set_ylabel("$r/R^*$ (-)")
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    fig.colorbar(sc, ax=axs, label="|M error| at the finest spec level (-)", shrink=0.9)
    return _save(plt, fig, stem)


def fig_mass(plt, results, keys, stem, spec_levels):
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.4), constrained_layout=True, sharey=False)
    cols = [MUTED, TEAL, INK, ACCENT, "#8AA5AC", "#B0B8BC"]
    for ax, key in zip(axs, keys):
        lv = [r for r in results[key]["levels"] if r.get("complete") and r["n"] in spec_levels]
        for c, r in zip(cols, lv):
            e = np.array(r["row_mdot_rel_err_vs_exact"]) * 100.0
            ax.plot(np.arange(len(e)) / max(len(e) - 1, 1), e, color=c, lw=1.0, label=f"n = {r['n']}")
        a, g = key.replace("alpha", "").replace("gamma", "").split("_")
        ax.text(0.98, 0.94, f"cone {a} deg, gamma {g}", transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color=INK)
        ax.set_xlabel("row index / last row index (-)")
        ax.set_ylabel("row mass-flow error vs exact (%)")
        ax.axhline(0, color=MUTED, lw=0.5)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
    axs[0].legend(frameon=False, loc="lower left", ncol=1)
    return _save(plt, fig, stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="13,25,49,97,193")
    ap.add_argument("--extra-levels", default="385")
    ap.add_argument("--configs", default="15:1.23,15:1.4,25:1.23,25:1.4")
    args = ap.parse_args()
    spec_levels = [int(v) for v in args.levels.split(",")]
    extra = [int(v) for v in args.extra_levels.split(",") if v]
    configs = [(float(a), float(b)) for a, b in (c.split(":") for c in args.configs.split(","))]
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results, finest = {}, {}
    for a, g in configs:
        key = f"alpha{a:g}_gamma{g:g}"
        lv, nets, mdot_ex, probes = run_config(a, g, spec_levels + extra)
        summ = summarize(lv, spec_levels)
        summ_extra = summarize(lv, spec_levels[1:] + extra) if extra else None
        results[key] = {"levels": lv, "exact_mdot": mdot_ex, "probes": probes, "summary_spec_levels": summ,
                        "summary_with_extra_level": summ_extra, "acceptance": acceptance(summ),
                        "acceptance_with_extra_level": acceptance(summ_extra) if summ_extra else None}
        finest[key] = nets[max(n for n in spec_levels if n in nets)]
        print(key, [(r["n"], r.get("complete"), round(r["wall_time_s"], 2)) for r in lv],
              results[key]["acceptance"], flush=True)
    all_complete = all(results[k]["summary_spec_levels"]["all_spec_levels_complete"] for k in results)
    acc_all = all(results[k]["acceptance"]["errors_all_nodes"]["monotone_decrease_M_and_theta"]
                  and results[k]["acceptance"]["errors_all_nodes"]["interior_order_ge_1.5"] for k in results)
    acc_excl = all(results[k]["acceptance"]["errors_excluding_exit_lip_row"]["monotone_decrease_M_and_theta"]
                   and results[k]["acceptance"]["errors_excluding_exit_lip_row"]["interior_order_ge_1.5"]
                   for k in results)
    moc14 = any(results[k]["acceptance"][sc]["MOC14_active_by_criterion"] for k in results
                for sc in ("errors_all_nodes", "errors_excluding_exit_lip_row"))
    if not all_complete:
        status = "INCOMPLETE"
    elif acc_all:
        status = "PASS"
    else:
        status = "FAIL"
    plt = _style()
    keys = list(results)
    figs = {
        "a_M": fig_convergence(plt, results, "M", "e05a_convergence_mach", spec_levels),
        "a_theta": fig_convergence(plt, results, "theta", "e05a_convergence_theta", spec_levels),
        "b": fig_map(plt, finest, [k for k in keys if k in ("alpha15_gamma1.23", "alpha25_gamma1.4")]
                     or keys[:2], "e05b_error_map_finest"),
        "c": fig_mass(plt, results, [k for k in keys if k in ("alpha15_gamma1.23", "alpha25_gamma1.4")]
                      or keys[:2], "e05c_row_massflow", spec_levels),
    }
    caption_common = ("Exact radial source flow in a straight cone (apex at origin, R* = 1), start line = exact "
                      "C- from the wall point at M = 1.5 to the axis (RK4), wall to M = 3 on the wall; host "
                      "lrekit.components.nozzle.solvers.bounded.march_bounded_net, arc None, dtheta_limit 0.5 deg, mass_tol 0.02, "
                      "on_invalid stop; n = start-line points.")
    result = {
        "id": "E05", "status": status,
        "status_reason": (
            "All spec levels complete. Acceptance on all nodes (spec): "
            + ("met" if acc_all else "NOT met (non-monotone max errors)")
            + ". The non-monotone errors sit in the single exit-lip closure row (special wall point at x_exit, "
              "moc_bounded 'beyond' branch); excluding that row, acceptance "
            + ("is met" if acc_excl else "is still not met") + ". MOC-14 by the 0.5-order criterion: "
            + ("active" if moc14 else "not active") + "."),
        "evidence_class": "verification (exact solution of the same Euler model)",
        "configuration": {
            "geometry": "apex at x = 0; cone half-angles 15 and 25 deg; R* = 1 (sonic sphere radius); start "
                        "wall point at R0 with M(R0) = 1.5; exit at the wall station where M = 3 "
                        "(x_exit = R_exit cos alpha)",
            "gas": "calorically perfect, gamma 1.23 and 1.4",
            "exact_field": "M(R) from A/A* = (R/R*)^2 (supersonic root of anderson Eq. 5.20, PDF 220 / printed 204), "
                           "theta = atan(r/x); A/A* = (R/R*)^2 is a derivation composed from sivells1978 PDF 16 / "
                           "printed 13 Eq. 17 and PDF 8, not a page quote (LR6)",
            "start_line": "exact C- (dx/dr = cot(theta - mu)) by RK4 in r from the wall point to the axis, "
                          "n points uniform in r, about 24000 RK4 steps in total; states from the exact field",
            "wall": "ConeWall: r = x tan(alpha), theta = alpha, x in [x0, x_exit]",
            "call": "march_bounded_net(start, wall, gamma, arc=None, dtheta_limit=radians(0.5), mass_tol=0.02, "
                    "on_invalid='stop')",
            "spec_levels": spec_levels, "supplementary_levels_beyond_spec": extra,
            "gci": "celik2008 with r = 2, Fs = 1.25 on three finest spec levels (equation page citations pending "
                   "LR4); probes: axis M midway between the exact axis feet of the start and exit-lip C-, wall M midway along the wall, exit-plane "
                   "mass-flow error (crop_to_exit_plane + exit_plane_mass_flow)",
            "exact_mass_flow": "rho* V* 2 pi R*^2 (1 - cos alpha) in the stagnation-normalised units of "
                               "calc_massflow_along_rrc",
        },
        "command": (".venv-jax/bin/python tests/experiments/nozzle/moc_ext_e05_source_flow_oracle.py "
                    + " ".join(sys.argv[1:])).strip(),
        "git_head": git_head(),
        "hashes": {"script": {str(Path(__file__).resolve().relative_to(ROOT)): sha256(Path(__file__))},
                   "raosim_imported": raosim_hashes(), "inputs": {}},
        "metrics": results,
        "hypotheses": {
            "H_E05_order_ge_1.5_including_axis": (
                "not supported on all nodes (exit-lip closure row non-convergent); supported once that row is "
                "excluded" if (not acc_all and acc_excl) else ("supported" if acc_all else "not supported")),
            "MOC14_classification": "active defect" if moc14 else "not active by the 0.5-order criterion",
        },
        "findings": [],
        "figures": [
            {"paths": figs["a_M"], "caption_facts": {
                "configuration": caption_common + " Mach-number error norms vs start-line spacing for four "
                                 "cone/gamma cases; dashed = exit-lip closure row excluded; grey band = level "
                                 "beyond the spec (supplementary).",
                "evidence_class": "verification",
                "not_established": "Accuracy of the throat kernel, KL start line, BDE inverse construction or "
                                   "curved walls; physical validity."}},
            {"paths": figs["a_theta"], "caption_facts": {
                "configuration": caption_common + " Flow-angle error norms (deg) vs spacing.",
                "evidence_class": "verification",
                "not_established": "As above."}},
            {"paths": figs["b"], "caption_facts": {
                "configuration": caption_common + " |M error| at every node of the finest spec level (n = "
                                 f"{max(spec_levels)}); squares outlined in accent = exit-lip closure row.",
                "evidence_class": "verification",
                "not_established": "As above."}},
            {"paths": figs["c"], "caption_facts": {
                "configuration": caption_common + " Per-row (RRC) mass-flow error against the exact source-flow "
                                 "mass flow, row index normalised by the last row.",
                "evidence_class": "verification",
                "not_established": "As above."}},
        ],
        "wall_time_s": time.time() - t0,
        "notes": [
            "Level n = 385 is beyond the spec (which asks for 193 if under 5 min); it is reported separately "
            "as supplementary and does not enter the acceptance decision.",
        ],
    }
    for key, r in results.items():
        g_ = r["summary_spec_levels"].get("gci_three_finest", {})
        ex = {"axis_M_probe": mach_of_R(r["probes"]["x_axis_probe"], float(key.split("gamma")[1])),
              "wall_M_probe": r["probes"]["x_wall_probe_exact_M"], "exit_plane_mdot": r["exact_mdot"]}
        for q, v in ex.items():
            if g_.get(q) and g_[q].get("phi_ext") is not None:
                g_[q]["exact"] = v
                g_[q]["phi_ext_minus_exact"] = g_[q]["phi_ext"] - v
                g_[q]["fine_minus_exact_rel"] = (g_[q]["phi"][0] - v) / v
                g_[q]["GCI_bounds_fine_error"] = abs(g_[q]["phi"][0] - v) / abs(g_[q]["phi"][0]) <= g_[q]["GCI_fine"]
        fin = [l for l in r["levels"] if l["n"] == max(spec_levels)][0]
        if fin.get("exit_lip_detail"):
            d = fin["exit_lip_detail"][0]
            result["findings"].append(
                f"{key} (n = {max(spec_levels)}): exit-lip row {d['row']} axis dM = {d['axis_dM']:.3e} at x = "
                f"{d['axis_x']:.3f}; its last interior point sits at r = {d['last_interior_r']:.3e}, "
                f"{d['last_interior_r'] / d['prev_row_last_interior_r']:.2f} of the previous row's "
                f"{d['prev_row_last_interior_r']:.3e} (special-row C+ from the previous axis point).")
    for _f in result["figures"]:
        _f.update({"script": str(Path(__file__).resolve().relative_to(ROOT)), "command": result["command"],
                   "git_head": result["git_head"], "status": result["status"],
                   "wall_time_s": result["wall_time_s"], "hashes": "see result.hashes (script, raosim, inputs)",
                   "file_sha256": {q: sha256(ROOT / q) for q in _f["paths"]}})
    (OUT / "result.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(status, result["status_reason"])
    return result


if __name__ == "__main__":
    main()
