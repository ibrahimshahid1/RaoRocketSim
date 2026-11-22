"""E10: fold/coalescence map of the current-chart TOP bell vs envelope and oblique-shock estimates.

Reproduces the t01 current_chart runs (validation/t01_coalescence_probe.json), maps the
same-family crossings at the fold, compares the fold with a planar simple-wave envelope
estimate from straight wall C- lines, and bounds the shock strength with calorically
perfect weak oblique-shock relations (anderson Eqs. 4.7-4.12, 4.17 PDF 151-152; 3.62-3.63
PDF 109). DIAGNOSTIC; changes nothing in raosim/.

    .venv-jax/bin/python tests/experiments/nozzle/moc_ext_e10_top_fold_envelope.py
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
OUT = ROOT / "builds/moc_extensions_plan_2026-11-21/opus_b/e10"
T01 = ROOT / ("docs/subsystems/S06-rao-moc-bvp-contour/audit_2026-11-19/evidence/validation/"
              "t01_coalescence_probe.json")

import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

import lrekit.components.nozzle.solvers.bounded as mb  # noqa: E402
from lrekit.components.nozzle.solvers.moc import solve_flowfield  # noqa: E402
from lrekit.components.nozzle.diagrams import _bde_same_family_crossings, _proper_segment_intersection  # noqa: E402
from lrekit.components.nozzle.geometry import bell_nozzle_contour  # noqa: E402
from lrekit.components.nozzle.solvers.wall import SplineWall  # noqa: E402

INK, TEAL, MUTED, ACCENT = "#19333C", "#176D76", "#51616B", "#C26A2E"
RT, EPS, GAMMA, LPCT = 0.02, 10.0, 1.2, 80.0
LEVELS = ((24, 0.5), (48, 0.25), (96, 0.125), (144, 0.0625))
MASS_TOL = 0.05


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raosim_hashes() -> dict:
    out = {}
    for name, mod in sorted(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if name.startswith("lrekit") and f and f.endswith(".py"):
            out[str(Path(f).resolve().relative_to(ROOT))] = sha256(Path(f))
    return out


def bezier_wall():
    c = bell_nozzle_contour(Rt=RT, epsilon=EPS, length_pct=LPCT, gamma=GAMMA,
                            theta_n_deg=None, theta_e_deg=None)
    xb, yb = np.asarray(c["x_bell"]), np.asarray(c["y_bell"])
    wall = SplineWall(xb, yb, slope_start=math.tan(math.radians(c["theta_n"])),
                      slope_end=math.tan(math.radians(c["theta_e"])))
    return wall, c


def run_level(wall, n, dth, on_invalid):
    calls = []
    orig = mb._build_row

    def wrapped(*a, **k):
        try:
            return orig(*a, **k)
        except mb._CellFold as exc:
            calls.append((a, k, exc))
            raise
    mb._build_row = wrapped
    try:
        t0 = time.time()
        res = solve_flowfield(RT, EPS, GAMMA, wall, n_char=n, dtheta_limit_deg=dth,
                              mass_tol=MASS_TOL, on_invalid=on_invalid)
        wt = time.time() - t0
    finally:
        mb._build_row = orig
    net = res["net"]
    crossing = None
    if calls:
        a, k, exc = calls[-1]
        built = orig(*a, **{**k, "orientation": 0.0})
        if built is not None:
            crossing = {"row": built[0], "parents": built[1], "fold_cell_index": exc.i,
                        "special": bool(a[2])}
    return res, net, crossing, wt, len(calls)


# --------------------------------------------------------------- crossing map
def _poly(row):
    return [np.array([p.x, p.r], dtype=float) for p in row]


def cplus_crossings(prev, new, parents):
    segs = []
    for k, ent in enumerate(parents):
        if ent and ent.get("cplus") is not None:
            _, ii = ent["cplus"]
            segs.append((np.array([prev[ii].x, prev[ii].r]), np.array([new[k].x, new[k].r])))
    out = []
    for (a, b), (c, d) in zip(segs[:-1], segs[1:]):
        p = _proper_segment_intersection(a, b, c, d)
        if p is not None:
            out.append((float(p[0]), float(p[1])))
    return out


def cell_areas(prev, new, special):
    i_last_prev = len(prev) - 1
    out = []
    for i in range(1, len(new) - 1):
        ii = min(i if special else i + 1, i_last_prev)
        if ii < 1:
            continue
        a = mb._signed_cell_area(prev[ii - 1], prev[ii], new[i], new[i - 1])
        e = math.hypot(prev[ii].x - prev[ii - 1].x, prev[ii].r - prev[ii - 1].r) * math.hypot(
            new[i].x - prev[ii].x, new[i].r - prev[ii].r)
        out.append((i, a / max(e, 1e-300), new[i].x, new[i].r))
    return out


def dist_point_polyline(p, poly):
    best = math.inf
    for a, b in zip(poly[:-1], poly[1:]):
        ab = b - a
        t = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0))
        best = min(best, float(np.linalg.norm(p - (a + t * ab))))
    return best


def crossing_angle_deg(rowA, rowB, p):
    def direction(row):
        best = None
        for a, b in zip(row[:-1], row[1:]):
            A, B = np.array([a.x, a.r]), np.array([b.x, b.r])
            ab = B - A
            t = float(np.clip(np.dot(p - A, ab) / np.dot(ab, ab), 0.0, 1.0))
            d = float(np.linalg.norm(p - (A + t * ab)))
            if best is None or d < best[0]:
                best = (d, math.degrees(math.atan2(ab[1], ab[0])))
        return best[1]
    a, b = direction(rowA), direction(rowB)
    return a, b, abs(a - b)


def interp_on_row(row, p):
    best, val = math.inf, None
    for a, b in zip(row[:-1], row[1:]):
        A, B = np.array([a.x, a.r]), np.array([b.x, b.r])
        ab = B - A
        t = 0.0 if not ab.any() else float(np.clip(np.dot(p - A, ab) / np.dot(ab, ab), 0.0, 1.0))
        d = float(np.linalg.norm(p - (A + t * ab)))
        if d < best:
            best, val = d, (a.M + t * (b.M - a.M), a.theta + t * (b.theta - a.theta))
    return val, best


# --------------------------------------------------------------- envelope
def envelope(net, crossing, wall, x_N):
    tol = 1e-12 * max(x_N, 1.0)
    W = [(row[0].x, row[0].r, row[0].theta, row[0].M) for row in net.rows[1:] if row[0].x >= x_N - tol]
    if crossing is not None:
        w = crossing["row"][0]
        if w.x > x_N:
            W.append((w.x, w.r, w.theta, w.M))
    W = np.array(sorted(set(W)))
    if len(W) < 2:
        return {"wall_states": W, "adjacent": [], "cusp": None, "coalescing_idx": [],
                "deflection_envelope_rad": 0.0, "n_wall_points": int(len(W))}
    x, r, th, M = W.T
    s = np.tan(th - np.arcsin(1.0 / M))
    adj = []
    for k in range(len(x) - 1):
        ds = s[k] - s[k + 1]
        if abs(ds) < 1e-15:
            continue
        xs = (r[k + 1] - r[k] + s[k] * x[k] - s[k + 1] * x[k + 1]) / ds
        rs = r[k] + s[k] * (xs - x[k])
        inside = xs > x[k + 1] and 0.0 <= rs <= wall.r(min(xs, wall.x_end)) and xs <= wall.x_end
        adj.append((k, xs, rs, inside))
    valid = [a for a in adj if a[3]]
    cusp = min(valid, key=lambda a: a[1]) if valid else None
    coalescing = []
    for k in range(1, len(x)):
        sj = s[:k]
        ds = sj - s[k]
        with np.errstate(divide="ignore", invalid="ignore"):
            xs = (r[k] - r[:k] + sj * x[:k] - s[k] * x[k]) / ds
        rs = r[:k] + sj * (xs - x[:k])
        ok = (xs > x[k]) & (xs <= wall.x_end) & (rs >= 0.0) & np.isfinite(xs)
        ok &= rs <= np.array([wall.r(v) if np.isfinite(v) and v <= wall.x_end else -1.0 for v in xs])
        if ok.any():
            coalescing.append(k)
    dtheta = [th[k - 1] - th[k] for k in coalescing]
    return {"wall_states": W, "adjacent": adj, "cusp": cusp, "coalescing_idx": coalescing,
            "deflection_envelope_rad": float(np.sum([d for d in dtheta if d > 0.0])),
            "n_wall_points": int(len(x))}


# --------------------------------------------------------------- oblique shock
def theta_beta(M1, beta, g):
    return math.atan(2.0 / math.tan(beta) * (M1 ** 2 * math.sin(beta) ** 2 - 1.0)
                     / (M1 ** 2 * (g + math.cos(2.0 * beta)) + 2.0))


def weak_shock(M1, theta, g):
    mu = math.asin(1.0 / M1)
    bs = np.linspace(mu + 1e-9, math.pi / 2 - 1e-9, 20001)
    tv = np.array([theta_beta(M1, b, g) for b in bs])
    i_max = int(np.argmax(tv))
    theta_max = float(tv[i_max])
    if theta > theta_max:
        return {"attached": False, "theta_max_deg": math.degrees(theta_max)}
    if theta <= 0.0:
        return {"attached": True, "beta_deg": math.degrees(mu), "p2_p1": 1.0, "p02_p01": 1.0,
                "ds_over_R": 0.0, "M2": M1, "theta_max_deg": math.degrees(theta_max)}
    beta = brentq(lambda b: theta_beta(M1, b, g) - theta, mu + 1e-12, float(bs[i_max]), xtol=1e-14)
    Mn1 = M1 * math.sin(beta)
    rho = (g + 1.0) * Mn1 ** 2 / ((g - 1.0) * Mn1 ** 2 + 2.0)
    p21 = 1.0 + 2.0 * g / (g + 1.0) * (Mn1 ** 2 - 1.0)
    Mn2 = math.sqrt((Mn1 ** 2 + 2.0 / (g - 1.0)) / (2.0 * g / (g - 1.0) * Mn1 ** 2 - 1.0))
    T21 = p21 / rho
    ds_R = g / (g - 1.0) * math.log(T21) - math.log(p21)
    return {"attached": True, "beta_deg": math.degrees(beta), "p2_p1": p21, "p02_p01": math.exp(-ds_R),
            "ds_over_R": ds_R, "M2": Mn2 / math.sin(beta - theta), "theta_max_deg": math.degrees(theta_max)}


# --------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "axes.labelsize": 8.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
                         "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "pdf.fonttype": 42})
    return plt


def _save(plt, fig, stem):
    paths = []
    for ext in ("pdf", "png"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(str(p.relative_to(ROOT)))
    plt.close(fig)
    return paths


def fig_net(plt, net, crossing, folds, env, contour, wall):
    from matplotlib.collections import LineCollection
    import matplotlib.colors as mcolors
    fig, (ax, az) = plt.subplots(1, 2, figsize=(6.5, 2.9), constrained_layout=True,
                                 gridspec_kw={"width_ratios": [1.9, 1.0]})
    segs, comp = [], []
    prev_th = None
    for j, row in enumerate(net.rows):
        xy = np.array([[p.x / RT, p.r / RT] for p in row])
        if j == 0:
            prev_th = row[0].theta
            ax.plot(xy[:, 0], xy[:, 1], color=MUTED, lw=0.6)
            continue
        segs.append(xy)
        comp.append(math.degrees(prev_th - row[0].theta))
        prev_th = row[0].theta
    comp = np.array(comp)
    vmax = max(float(np.percentile(np.abs(comp), 99)), 1e-6)
    lc = LineCollection(segs, cmap="cividis", norm=mcolors.Normalize(vmin=-vmax, vmax=vmax), lw=0.25)
    lc.set_array(comp)
    ax.add_collection(lc)
    xa = np.array(contour["x"]) / RT
    ya = np.array(contour["y"]) / RT
    m = xa >= 0
    ax.plot(xa[m], ya[m], color=INK, lw=1.0)
    cx = None
    if crossing is not None:
        cx = np.array([[p.x / RT, p.r / RT] for p in crossing["row"]])
        ax.plot(cx[:, 0], cx[:, 1], color=ACCENT, lw=0.9, label="folding C- (first bell row, rebuilt)")
    mk = ["o", "s", "^", "D"]
    for (lab, f), m_ in zip(folds, mk):
        ax.plot(f[0] / RT, f[1] / RT, m_, ms=4, mfc="white", mec=INK, mew=0.8, label=lab)
    if env["cusp"] is not None:
        ax.plot(env["cusp"][1] / RT, env["cusp"][2] / RT, "*", ms=8, color=ACCENT, mec=INK, mew=0.4,
                label="envelope estimate")
    N = contour["N"]
    ax.plot(N[0] / RT, N[1] / RT, "P", ms=5, color=TEAL, label="N (bell start)")
    xmax = max(max(p.x for p in r) for r in net.rows) / RT
    ax.set_xlim(-0.1, xmax * 1.01)
    ax.set_ylim(0, math.sqrt(EPS) * 1.05)
    ax.set_xlabel("$x/R_t$ (-)")
    ax.set_ylabel("$r/R_t$ (-)")
    cb = fig.colorbar(lc, ax=ax, shrink=0.85, pad=0.01, location="bottom", aspect=40)
    cb.set_label("wall-angle change per C- (deg; < 0 = expansion)")
    ax.legend(frameon=False, loc="upper left", fontsize=6.3, ncol=2)
    f = np.array(folds[-1][1]) / RT
    if cx is not None:
        lx = np.array([[p.x / RT, p.r / RT] for p in net.rows[-1]])
        px = np.array([[p.x / RT, p.r / RT] for p in net.rows[-2]])
        if np.all(np.diff(lx[:, 0]) > 0) and np.all(np.diff(cx[:, 0]) > 0) and np.all(np.diff(px[:, 0]) > 0):
            xs = np.linspace(max(lx[0, 0], cx[0, 0], px[0, 0]), min(lx[-1, 0], cx[-1, 0], px[-1, 0]), 4000)
            gap_fold = np.interp(xs, cx[:, 0], cx[:, 1]) - np.interp(xs, lx[:, 0], lx[:, 1])
            gap_prev = np.interp(xs, lx[:, 0], lx[:, 1]) - np.interp(xs, px[:, 0], px[:, 1])
            az.plot(xs, gap_fold * 1e3, color=ACCENT, lw=1.0, label="folding minus last stored C-")
            az.plot(xs, gap_prev * 1e3, color=INK, lw=0.8, ls="--", label="last minus second-last stored C-")
            az.axhline(0.0, color=MUTED, lw=0.5)
            az.axvline(f[0], color=MUTED, lw=0.5, ls=":")
            az.legend(frameon=False, fontsize=6.3, loc="upper right")
    az.set_xlabel("$x/R_t$ (-)")
    az.set_ylabel(r"vertical gap between C- ($10^{-3} R_t$)")
    for a_ in (ax, az):
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
    return _save(plt, fig, "e10a_net_fold_envelope")


def fig_drift(plt, tab):
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.4), constrained_layout=True)
    for oi, ls, mk in (("refine", "-", "o"), ("stop", "--", "s")):
        rows = [t for t in tab if t["on_invalid"] == oi and t["fold"] is not None]
        h = np.array([t["dtheta_deg"] for t in rows])
        fx = np.array([t["fold"][0] / RT for t in rows])
        fr = np.array([t["fold"][1] / RT for t in rows])
        lab = f"on_invalid = '{oi}'" + (" (t01 runs)" if oi == "refine" else "")
        axs[0].semilogx(h, fx, ls=ls, marker=mk, color=INK if oi == "refine" else MUTED, ms=4, lw=1,
                        mfc="white" if oi == "stop" else INK, label=lab)
        axs[1].semilogx(h, fr, ls=ls, marker=mk, color=TEAL if oi == "refine" else MUTED, ms=4, lw=1,
                        mfc="white" if oi == "stop" else TEAL, label=lab)
    axs[0].set_ylabel("fold $x/R_t$ (-)")
    axs[1].set_ylabel("fold $r/R_t$ (-)")
    for ax in axs:
        ax.set_xlabel(r"wall-turning limit $\Delta\theta_B$ (deg); $n$ = 24, 48, 96, 144")
        ax.set_xticks([d for _, d in LEVELS])
        ax.set_xticklabels([f"{d:g}" for _, d in LEVELS])
        ax.minorticks_off()
        ax.invert_xaxis()
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.legend(frameon=False, fontsize=7)
    return _save(plt, fig, "e10b_fold_vs_spacing")


def fig_strength(plt, M1, est, g):
    fig, ax = plt.subplots(figsize=(3.2, 2.5), constrained_layout=True)
    th = np.linspace(0.0, math.radians(20.0), 201)
    vals = [weak_shock(M1, t, g) for t in th]
    pp = np.array([v["p02_p01"] if v["attached"] else np.nan for v in vals])
    ax.plot(np.degrees(th), pp, color=INK, lw=1.1)
    ax.axhline(0.99, color=MUTED, lw=0.6, ls=":")
    for (lab, d, c) in est:
        v = weak_shock(M1, d, g)
        ax.axvline(math.degrees(d), color=c, lw=0.9, ls="--")
        if v["attached"]:
            ax.plot(math.degrees(d), v["p02_p01"], "o", ms=4, color=c)
        ax.text(math.degrees(d), 0.86 if c == ACCENT else 0.9, " " + lab, color=c, fontsize=6.5,
                rotation=90, va="bottom")
    ax.set_xlabel("assumed flow deflection (deg)")
    ax.set_ylabel(r"$p_{02}/p_{01}$ (-)")
    ax.set_ylim(0.85, 1.002)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    return _save(plt, fig, "e10c_total_pressure_vs_deflection")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    wall, contour = bezier_wall()
    x_N = float(contour["N"][0])
    t01 = json.loads(T01.read_text())
    ref = {r["n"]: r for r in t01["runs"] if r["label"] == "current_chart"}
    tab, nets = [], {}
    for oi in ("refine", "stop"):
        for n, dth in LEVELS:
            res, net, crossing, wt, nfold = run_level(wall, n, dth, oi)
            f = net.fold_location
            t = {"on_invalid": oi, "n": n, "dtheta_deg": dth, "stop_reason": net.stop_reason,
                 "complete": bool(net.complete), "n_rows": net.n_rows, "fold": f, "wall_time_s": wt,
                 "fold_raising_attempts": nfold, "n_refined": net.n_refined,
                 "containment_violations": net.containment_violations}
            if oi == "refine" and n in ref and f is not None:
                t["t01_fold"] = ref[n]["fold"]
                t["abs_dev_m"] = [abs(f[0] - ref[n]["fold"][0]), abs(f[1] - ref[n]["fold"][1])]
                t["t01_n_rows"] = ref[n]["n_rows"]
            tab.append(t)
            nets[(oi, n)] = (net, crossing)
            if crossing is not None and f is not None:
                la, lb, ang = crossing_angle_deg(net.rows[-1], crossing["row"], np.array(f))
                t["crossing_angle_deg"] = ang
                t["last_row_dir_deg"], t["folding_row_dir_deg"] = la, lb
                t["last_stored_wall_x_m"] = float(net.rows[-1][0].x)
                t["last_stored_wall_is_N"] = abs(net.rows[-1][0].x - float(bezier_wall()[1]["N"][0])) <= 1e-12
                t["folding_row_wall"] = {"x": crossing["row"][0].x, "theta_deg": math.degrees(crossing["row"][0].theta),
                                         "M": crossing["row"][0].M, "special": crossing["special"]}
                t["n_stored_wall_points_beyond_N"] = sum(
                    1 for r_ in net.rows[1:] if r_[0].x > float(bezier_wall()[1]["N"][0]) + 1e-12)
            print(oi, n, net.stop_reason, net.n_rows, f, round(wt, 2), flush=True)
    repro_ok = all(t["abs_dev_m"][0] <= 1e-9 and t["abs_dev_m"][1] <= 1e-9 and t["n_rows"] == t["t01_n_rows"]
                   for t in tab if t["on_invalid"] == "refine")
    net, crossing = nets[("refine", LEVELS[-1][0])]
    fold = np.array(net.fold_location)

    last = net.rows[-12:]
    rrc_stored = _bde_same_family_crossings(last)["rrc"]
    rrc_fold = _bde_same_family_crossings([net.rows[-1], crossing["row"]])["rrc"] if crossing else []
    cpl = cplus_crossings(net.rows[-1], crossing["row"], crossing["parents"]) if crossing else []
    areas_row2 = [a for _, a, _, _ in cell_areas(net.rows[1], net.rows[2], net.special[2])]
    orient = math.copysign(1.0, float(np.median([a for a in areas_row2 if a != 0.0])))
    area_tab = []
    for j in range(net.n_rows - 6, net.n_rows):
        ca = cell_areas(net.rows[j - 1], net.rows[j], net.special[j])
        area_tab.append({"row": j, "min_oriented_area_norm": float(min(orient * a for _, a, _, _ in ca))})
    ca_fold = cell_areas(net.rows[-1], crossing["row"], crossing["special"]) if crossing else []
    nonpos = [{"i": i, "x": x, "r": r, "oriented_area_norm": orient * a} for i, a, x, r in ca_fold
              if orient * a <= 1e-12]
    area_tab.append({"row": "folding (rebuilt)", "min_oriented_area_norm":
                     float(min(orient * a for _, a, _, _ in ca_fold)) if ca_fold else None,
                     "n_nonpositive": len(nonpos)})
    gap = None
    if crossing is not None:
        lx = np.array([[p.x, p.r] for p in net.rows[-1]])
        cx_ = np.array([[p.x, p.r] for p in crossing["row"]])
        if np.all(np.diff(lx[:, 0]) > 0) and np.all(np.diff(cx_[:, 0]) > 0):
            xs = np.linspace(max(lx[0, 0], cx_[0, 0]), min(lx[-1, 0], cx_[-1, 0]), 20001)
            gfold = np.interp(xs, cx_[:, 0], cx_[:, 1]) - np.interp(xs, lx[:, 0], lx[:, 1])
            zc = np.where(np.diff(np.sign(gfold)) != 0)[0]
            gap = {"gap_near_wall_over_Rt": float(gfold[0] / RT), "gap_at_axis_end_over_Rt": float(gfold[-1] / RT),
                   "zero_crossings_x_over_Rt": [float(xs[k] / RT) for k in zc],
                   "definition": "r(folding C-) - r(last stored C-) at equal x, n = 144"}
    near = min(net.rows[-1], key=lambda p: math.hypot(p.x - fold[0], p.r - fold[1]))
    h_loc = dist_point_polyline(np.array([near.x, near.r]), _poly(net.rows[-2]))

    env_by_level = {}
    for n, _ in LEVELS:
        e = envelope(nets[("refine", n)][0], nets[("refine", n)][1], wall, x_N)
        env_by_level[n] = e
    env = env_by_level[LEVELS[-1][0]]
    cusp = env["cusp"]
    d_cusp = float(math.hypot(cusp[1] - fold[0], cusp[2] - fold[1])) if cusp else None

    (M1, th1), dnear = interp_on_row(net.rows[-1], fold)
    defl_upper = math.radians(contour["theta_n"] - contour["theta_e"])
    defl_env = env["deflection_envelope_rad"]
    sh_upper = weak_shock(M1, defl_upper, GAMMA)
    sh_env = weak_shock(M1, defl_env, GAMMA)
    M_at_N = float(net.rows[-1][0].M)

    rf = [t for t in tab if t["on_invalid"] == "refine"]
    fr = np.array([t["fold"][1] for t in rf])
    fx = np.array([t["fold"][0] for t in rf])
    r_sp, x_sp = float(fr.max() - fr.min()), float(fx.max() - fx.min())
    crit_rel = (r_sp / float(fr.mean())) < 0.25 * (x_sp / float(fx.mean()))
    crit_hloc = r_sp < h_loc < x_sp
    H1 = {"r_range_m": r_sp, "x_range_m": x_sp, "h_loc_finest_m": h_loc,
          "r_rel_spread": r_sp / float(fr.mean()), "x_rel_spread": x_sp / float(fx.mean()),
          "r_change_finest_pair_m": float(abs(fr[-1] - fr[-2])), "x_change_finest_pair_m": float(abs(fx[-1] - fx[-2])),
          "r_over_Rt": (fr / RT).tolist(), "x_mm": (fx * 1e3).tolist(),
          "crossing_angle_deg_by_level": [t.get("crossing_angle_deg") for t in rf],
          "criterion_relative": "relative r spread below a quarter of the relative x spread: " + str(crit_rel),
          "criterion_h_loc": "r spread below the finest local C- spacing h_loc and x spread above it: " + str(crit_hloc),
          "verdict": ("supported" if (crit_rel and crit_hloc) else
                      ("partly supported: r is far more stable than x, but its spread exceeds the finest local "
                       "C- spacing, so r is not shown converged" if crit_rel else "not supported")),
          "explanation": "the folding C- (first bell row) crosses the C- from N at a grazing angle "
                         "(see crossing_angle_deg_by_level), so a small change in either line moves the crossing "
                         "mostly in x"}
    H2 = {"wall_states_used": env["wall_states"].tolist(),
          "limitation": "the forward net stops at the first bell row at every level (last stored wall point = N), "
                        "so the only wall states downstream of N are N itself and the first bell wall point of "
                        "the rebuilt folding row: one adjacent pair",
          "adjacent_pair_intersections_any_location": [
              {"k": a[0], "x_m": a[1], "r_m": a[2], "inside_nozzle": a[3]} for a in env["adjacent"]],
          "cusp_xr_m": [cusp[1], cusp[2]] if cusp else None, "fold_xr_m": fold.tolist(),
          "distance_m": d_cusp, "h_loc_m": h_loc,
          "distance_over_h_loc": (d_cusp / h_loc) if d_cusp is not None else None,
          "verdict": ("undetermined (degenerate: one wall pair; request filed)" if env["n_wall_points"] <= 2 else
                      ("supported" if (d_cusp is not None and d_cusp <= h_loc) else "not supported")),
          "verdict_degenerate_pair": ("within h_loc" if (d_cusp is not None and d_cusp <= h_loc) else
                                      "not within h_loc"),
          "cusp_by_level": {str(n): ([e["cusp"][1], e["cusp"][2]] if e["cusp"] else None)
                            for n, e in env_by_level.items()}}
    H3 = {"M1_at_fold": M1, "theta_at_fold_deg": math.degrees(th1), "M_at_N": M_at_N,
          "deflection_upper_deg": math.degrees(defl_upper), "deflection_envelope_deg": math.degrees(defl_env),
          "n_coalescing_wall_C-": len(env["coalescing_idx"]), "n_bell_wall_points": env["n_wall_points"],
          "upper_bound": sh_upper, "envelope_consistent": sh_env,
          "verdict": ("undetermined (envelope-consistent deflection is degenerate: one coalescing wall C-; "
                      "request filed)" if env["n_wall_points"] <= 2 else
                      ("supported" if sh_env.get("attached") and sh_env["p02_p01"] >= 0.99 else "not supported")),
          "upper_bound_meets_0.99": bool(sh_upper.get("attached") and sh_upper["p02_p01"] >= 0.99)}

    plt = _plt()
    folds = [(f"fold n = {t['n']}", t["fold"]) for t in rf]
    figs = {"a": fig_net(plt, net, crossing, folds, env, contour, wall), "b": fig_drift(plt, tab),
            "c": fig_strength(plt, M1, [("upper bound", defl_upper, INK),
                                        ("envelope-consistent" + (" (degenerate)" if env["n_wall_points"] <= 2
                                                                  else ""), defl_env, ACCENT)], GAMMA)}
    degenerate = env["n_wall_points"] <= 2
    status = "FAIL" if not repro_ok else ("INCOMPLETE" if degenerate else "PASS-DIAGNOSTIC")
    cap = ("Current-chart TOP Bezier bell (Rt 0.02 m, eps 10, 80 %, gamma 1.2, theta_n 25.74, theta_e 10.49 deg, "
           "rounded-throat length 0.12992 m, Rd 0.382 Rt), KL start line, host bounded net via "
           "lrekit.components.nozzle.solvers.moc.solve_flowfield with mass_tol 0.05 and its default on_invalid='refine' (the t01 runs).")
    result = {
        "id": "E10", "status": status,
        "evidence_class": "neither (diagnostic); reproduction step is verification",
        "configuration": {
            "case": cap, "levels": [{"n": n, "dtheta_deg": d} for n, d in LEVELS],
            "on_invalid": "both 'refine' (solve_flowfield default; reproduces t01 exactly) and 'stop' (as written "
                          "in the spec); H1-H3 use the 'refine' nets",
            "crossing_map": "adjacent-RRC crossings over the last 12 stored rows and between the last stored row "
                            "and the folding row, rebuilt by re-calling moc_bounded._build_row with the fold "
                            "audit off (orientation 0) for that one row only; C+ crossings from the explicit "
                            "parent map; signed cell areas normalised by edge products",
            "h_loc": "distance from the last stored row's node nearest the fold to the previous row polyline",
            "envelope": "finest-level wall states downstream of N; straight C- lines slope tan(theta_w - mu_w); "
                        "adjacent-pair intersections downstream of both origins, 0 <= r <= r_wall, x <= x_E; "
                        "earliest = smallest x (courant1944 envelope; inouye1965 shock start, as cited by the spec)",
            "strength": "weak oblique shock, calorically perfect, gamma 1.2 (anderson Eq. 4.17 PDF 152 / printed 136; "
                        "Eqs. 4.7-4.12 PDF 151 / printed 135; Eqs. 3.62-3.63 PDF 109 / printed 93); M1 interpolated "
                        "on the last stored row at the fold; deflections: (i) theta_n - theta_e, (ii) sum of positive "
                        "wall-angle decrements of wall C- whose straight extension crosses an earlier one before "
                        "the exit plane inside the domain",
        },
        "command": ".venv-jax/bin/python tests/experiments/nozzle/moc_ext_e10_top_fold_envelope.py",
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                   text=True).stdout.strip(),
        "hashes": {"script": {str(Path(__file__).resolve().relative_to(ROOT)): sha256(Path(__file__))},
                   "raosim_imported": raosim_hashes(),
                   "inputs": {str(T01.relative_to(ROOT)): sha256(T01)}},
        "metrics": {
            "reproduction": {"pass_1e-9_m_and_rows": repro_ok, "table": tab},
            "crossing_map_finest": {
                "rrc_crossings_last_12_stored_rows": rrc_stored,
                "rrc_crossings_last_stored_vs_folding_row": rrc_fold,
                "cplus_crossings_folding_strip": cpl,
                "cell_area_table": area_tab, "nonpositive_cells_folding_row": nonpos,
                "folding_row_special": crossing["special"] if crossing else None,
                "folding_vs_last_stored_gap": gap,
                "fold_cell_index": crossing["fold_cell_index"] if crossing else None,
                "first_crossing_locus_refine": [t["fold"] for t in rf],
                "first_crossing_locus_stop": [t["fold"] for t in tab if t["on_invalid"] == "stop"],
            },
            "envelope_finest": {"cusp": cusp, "n_adjacent_pairs": len(env["adjacent"]),
                                "n_inside": sum(1 for a in env["adjacent"] if a[3])},
        },
        "hypotheses": {"H1": H1, "H2": H2, "H3": H3},
        "figures": [
            {"paths": figs["a"], "caption_facts": {
                "configuration": cap + " Finest level (n 144, 0.0625 deg): RRCs coloured by the wall-angle "
                                 "decrement at their wall point; fold points of the four levels; envelope estimate; "
                                 "N; the folding C- rebuilt with the fold audit off.",
                "evidence_class": "neither (diagnostic)",
                "not_established": "The physical internal-shock position; that the fold is a shock rather than "
                                   "under-resolution."}},
            {"paths": figs["b"], "caption_facts": {
                "configuration": cap + " Fold location vs wall-turning limit for on_invalid 'refine' (t01) and "
                                 "'stop'.",
                "evidence_class": "neither (diagnostic); the 'refine' points reproduce t01 (verification)",
                "not_established": "Resolution convergence of the fold; a physical shock location."}},
            {"paths": figs["c"], "caption_facts": {
                "configuration": f"Weak oblique-shock total-pressure ratio at M1 = {M1:.3f} (gamma 1.2) vs assumed "
                                 "deflection; dashed: upper bound theta_n - theta_e and the envelope-consistent "
                                 "deflection.",
                "evidence_class": "neither (diagnostic, calorically perfect straight-shock bound)",
                "not_established": "Shock curvature, axisymmetric strengthening, reflections, viscous effects; "
                                   "no open numeric TOP internal-shock data exist (ostlund PDF 26, ruf2010)."}},
        ],
        "wall_time_s": time.time() - t0,
        "findings": [
            "At every level and for both on_invalid settings the last stored wall point is N itself; the march "
            "folds on the first C- from the bell (no stored wall point lies on the bell). With 'refine' the "
            "folding row is a special (inserted) row; with 'stop' it is a raw row; both fold.",
            "The folding C- meets the C- from N at a grazing angle (0.50, 0.15, 0.12, 0.018 deg at n 24, 48, 96, "
            "144 with 'refine'), which is why the fold drifts in x but hardly in r.",
            "The spec's envelope estimate (H2) and envelope-consistent deflection (H3 ii) need wall states on the "
            "bell; only one pair (N, first bell point) exists and its straight C- extensions meet outside the "
            "nozzle, so both are undetermined pending a decision (outbox request).",
        ],
        "notes": [
            "The spec lists on_invalid 'stop' but also requires exact t01 reproduction; t01 used solve_flowfield's "
            "default 'refine'. Both were run; 'stop' moves the fold by up to about 1 mm at n 24.",
        ],
    }
    for _f in result["figures"]:
        _f.update({"script": str(Path(__file__).resolve().relative_to(ROOT)), "command": result["command"],
                   "git_head": result["git_head"], "status": result["status"],
                   "wall_time_s": result["wall_time_s"], "hashes": "see result.hashes (script, raosim, inputs)",
                   "file_sha256": {q: sha256(ROOT / q) for q in _f["paths"]}})
    (OUT / "result.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"status": status, "H1": H1["verdict"], "H2": H2, "H3": {k: v for k, v in H3.items()}},
                     indent=1, default=float))


if __name__ == "__main__":
    main()
