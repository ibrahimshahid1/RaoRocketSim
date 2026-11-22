"""E12: Taylor and topology-event test of the JAX throat-kernel derivative
d(BD outputs)/d(theta_B).  Opt-in experiment script; read-only use of
lrekit.numerics.jax.moc_kernel.march_kernel and lrekit.numerics.jax.theta_b_solve.

.venv-jax/bin/python tests/experiments/nozzle/moc_ext_e12_jax_kernel_taylor.py
"""
from __future__ import annotations

from lrekit.paths import repository_root as _repository_root

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time

import numpy as np

ROOT = str(_repository_root())
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import lrekit.numerics.jax  # noqa: E402,F401
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from lrekit.numerics.jax import assembly  # noqa: E402
from lrekit.numerics.jax.moc_kernel import march_kernel  # noqa: E402
from lrekit.numerics.jax.theta_b_solve import _n_rows_bound, start_line_from_kernel  # noqa: E402
from lrekit.components.nozzle.solvers.nasa import build_kernel  # noqa: E402

OUT_DEFAULT = os.path.join(ROOT, "builds", "moc_extensions_plan_2026-11-21", "opus_b", "e12")
RT, RD, GAMMA, N_KERNEL, RU = 1.0, 0.382, 1.23, 24, 1.5
DTL = math.radians(0.25)
TB_LO_DEG, TB_HI_DEG = 27.0, 28.5
ROW_MARGIN = 32
H_DEG = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
TAYLOR_TARGETS_DEG = [27.25, 27.75, 28.25]
NOISE_FLOOR_FACTOR = 100.0
JUMP_MAD_K = 20.0
J_NAMES = ["J1_mdot_BD", "J2_M_B", "J3_x_D_axis"]

INK, TEAL, MUTED, ACCENT = "#19333C", "#176D76", "#51616B", "#C26A2E"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_arrays(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return h.hexdigest()


def git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def raosim_source_hashes():
    out = {}
    for name, mod in sorted(sys.modules.items()):
        if not (name == "lrekit" or name.startswith("lrekit.")):
            continue
        f = getattr(mod, "__file__", None)
        if f and f.endswith(".py") and os.path.exists(f):
            out[os.path.relpath(f, ROOT)] = sha256_file(f)
    return out


def make_functions(sl, n_rows):
    def outputs(tb):
        m = march_kernel(sl, tb, RT, RD, GAMMA, n_rows=n_rows, n_corr=40, dtheta_limit=DTL)
        bd = m.bd
        j1 = assembly._polyline_mass_flux(bd.x, bd.r, bd.M, bd.theta, GAMMA, assembly._CE_M_FLOOR)
        J = jnp.stack([j1, bd.M[0], bd.x[m.bd_axis_idx]])
        aux = dict(reached=m.reached_wall, axis_idx=m.bd_axis_idx,
                   row_axis_idx=m.row_axis_idx, wall_theta=m.wall_theta)
        return J, aux

    def with_tangent(tb):
        J, dJ, aux = jax.jvp(outputs, (tb,), (jnp.ones_like(tb),), has_aux=True)
        return J, dJ, aux

    return outputs, with_tangent


def pattern_of(theta_b, row_axis_idx, wall_theta):
    wt = np.asarray(wall_theta)
    ax = np.asarray(row_axis_idx)
    hit = np.nonzero(wt[1:] >= theta_b - 1e-9)[0]
    if hit.size == 0:
        return dict(n_active=None, flags="", n_special=None, n_raw=None, pre_clamp_last_deg=None)
    n_active = int(hit[0]) + 1
    flags = "".join("S" if ax[k] - ax[k - 1] == 1 else "R" for k in range(1, n_active + 1))
    return dict(n_active=n_active, flags=flags, n_special=flags.count("S"), n_raw=flags.count("R"),
                pre_clamp_last_deg=float(np.degrees(wt[n_active - 1])),
                signature=hashlib.sha256(f"{n_active}:{flags}".encode()).hexdigest()[:16])


def robust_threshold(stat):
    med = float(np.median(stat))
    mad = float(np.median(np.abs(stat - med)))
    return med + JUMP_MAD_K * 1.4826 * mad, med, mad


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7.5, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK, "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
    })
    return plt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-points", type=int, default=301)
    ap.add_argument("--chunk", type=int, default=43)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()
    timings = {}

    t0 = time.time()
    k_hi = build_kernel(RT, RD, math.radians(TB_HI_DEG), GAMMA, N_KERNEL, starting_line_method="kliegel_levine",
                        max_rrcs=500, dtheta_limit=DTL, Ru=RU)
    k_lo = build_kernel(RT, RD, math.radians(TB_LO_DEG), GAMMA, N_KERNEL, starting_line_method="kliegel_levine",
                        max_rrcs=500, dtheta_limit=DTL, Ru=RU)
    timings["host_kernel_build_s"] = time.time() - t0
    sl = start_line_from_kernel(k_hi)
    sl_lo = start_line_from_kernel(k_lo)
    sl_identical = all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(sl, sl_lo))
    n_rows = _n_rows_bound(math.radians(TB_HI_DEG), DTL) + ROW_MARGIN

    outputs, with_tangent = make_functions(sl, n_rows)

    theta_deg = np.linspace(TB_LO_DEG, TB_HI_DEG, args.n_points)
    theta_rad = np.radians(theta_deg)
    n_chunks = int(math.ceil(args.n_points / args.chunk))
    pad = n_chunks * args.chunk - args.n_points
    tb_pad = np.concatenate([theta_rad, np.full(pad, theta_rad[-1])])

    sweep_fn = jax.jit(jax.vmap(with_tangent))
    t0 = time.time()
    sweep_c = sweep_fn.lower(jnp.asarray(tb_pad[:args.chunk])).compile()
    timings["sweep_jit_compile_s"] = time.time() - t0
    t0 = time.time()
    Js, dJs, reached, axis_idx, row_ax, wall_th = [], [], [], [], [], []
    for c in range(n_chunks):
        J, dJ, aux = sweep_c(jnp.asarray(tb_pad[c * args.chunk:(c + 1) * args.chunk]))
        jax.block_until_ready(J)
        Js.append(np.asarray(J)); dJs.append(np.asarray(dJ))
        reached.append(np.asarray(aux["reached"])); axis_idx.append(np.asarray(aux["axis_idx"]))
        row_ax.append(np.asarray(aux["row_axis_idx"])); wall_th.append(np.asarray(aux["wall_theta"]))
    timings["sweep_eval_s"] = time.time() - t0
    timings["sweep_eval_per_point_s"] = timings["sweep_eval_s"] / (n_chunks * args.chunk)
    n = args.n_points
    J = np.concatenate(Js)[:n]; dJ = np.concatenate(dJs)[:n]
    reached = np.concatenate(reached)[:n]; axis_idx = np.concatenate(axis_idx)[:n]
    row_ax = np.concatenate(row_ax)[:n]; wall_th = np.concatenate(wall_th)[:n]

    patterns = [pattern_of(theta_rad[i], row_ax[i], wall_th[i]) for i in range(n)]
    n_incomplete = int(np.sum(~reached))
    valid = reached.copy()

    sig = [p.get("signature") for p in patterns]
    windows = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sig[j + 1] == sig[i] and valid[j + 1] == valid[i]:
            j += 1
        windows.append(dict(i0=i, i1=j, theta_lo_deg=float(theta_deg[i]), theta_hi_deg=float(theta_deg[j]),
                            n_active=patterns[i]["n_active"], n_special=patterns[i]["n_special"],
                            n_raw=patterns[i]["n_raw"], pre_clamp_last_deg=patterns[i]["pre_clamp_last_deg"],
                            signature=sig[i], valid=bool(valid[i])))
        i = j + 1

    pattern_change = np.array([sig[i] != sig[i + 1] for i in range(n - 1)])
    rowcount_change = np.array([patterns[i]["n_active"] != patterns[i + 1]["n_active"] for i in range(n - 1)])
    interval_valid = valid[:-1] & valid[1:]
    dt = np.diff(theta_rad)

    jump_report = {}
    for q, name in enumerate(J_NAMES):
        r_val = np.abs(J[1:, q] - J[:-1, q] - dt * 0.5 * (dJ[1:, q] + dJ[:-1, q]))
        r_der = np.abs(dJ[1:, q] - dJ[:-1, q])
        thr_v, med_v, mad_v = robust_threshold(r_val[interval_valid])
        thr_d, med_d, mad_d = robust_threshold(r_der[interval_valid])
        jv = (r_val > thr_v) & interval_valid
        jd = (r_der > thr_d) & interval_valid
        jumps = jv | jd
        unexplained = np.nonzero(jumps & ~pattern_change)[0]
        silent = np.nonzero(pattern_change & ~jumps & interval_valid)[0]
        jump_report[name] = dict(
            value_residual_threshold=thr_v, value_residual_median=med_v, value_residual_mad=mad_v,
            derivative_jump_threshold=thr_d, derivative_jump_median=med_d, derivative_jump_mad=mad_d,
            n_value_jumps=int(jv.sum()), n_derivative_jumps=int(jd.sum()), n_jumps=int(jumps.sum()),
            n_pattern_changes=int(pattern_change.sum()), n_rowcount_changes=int(rowcount_change.sum()),
            n_jumps_without_pattern_change=int(unexplained.size),
            jumps_without_pattern_change_theta_deg=[float(theta_deg[k]) for k in unexplained],
            n_pattern_changes_without_detected_jump=int(silent.size),
            pattern_changes_without_detected_jump_theta_deg=[float(theta_deg[k]) for k in silent],
            jump_intervals_theta_deg=[[float(theta_deg[k]), float(theta_deg[k + 1])] for k in np.nonzero(jumps)[0]],
            max_value_residual_inside_windows=float(np.max(r_val[interval_valid & ~pattern_change])),
            min_value_residual_at_pattern_changes=(float(np.min(r_val[interval_valid & pattern_change]))
                                                   if np.any(interval_valid & pattern_change) else None),
            _r_val=r_val, _r_der=r_der, _jumps=jumps,
        )

    single = jax.jit(outputs)
    single_t = jax.jit(with_tangent)
    t0 = time.time()
    single_c = single.lower(jnp.float64(theta_rad[0])).compile()
    single_t_c = single_t.lower(jnp.float64(theta_rad[0])).compile()
    timings["taylor_jit_compile_s"] = time.time() - t0

    cand = [w for w in windows if w["valid"] and (w["i1"] - w["i0"]) >= 2]
    chosen = []
    for tgt in TAYLOR_TARGETS_DEG:
        best = min((w for w in cand if w not in chosen),
                   key=lambda w: abs(0.5 * (w["theta_lo_deg"] + w["theta_hi_deg"]) - tgt))
        chosen.append(best)

    t0 = time.time()
    taylor = []
    for w in chosen:
        t_deg = float(theta_deg[w["i0"] + 1])
        t = math.radians(t_deg)
        J0, dJ0, aux0 = single_t_c(jnp.float64(t))
        J0 = np.asarray(J0); dJ0 = np.asarray(dJ0)
        p0 = pattern_of(t, aux0["row_axis_idx"], aux0["wall_theta"])
        rows = []
        for hd in H_DEG:
            h = math.radians(hd)
            Jh, auxh = single_c(jnp.float64(t + h))
            Jh = np.asarray(Jh)
            ph = pattern_of(t + h, auxh["row_axis_idx"], auxh["wall_theta"])
            E = np.abs(Jh - J0 - h * dJ0)
            rows.append(dict(h_deg=hd, h_rad=h, E=[float(e) for e in E],
                             same_window=bool(ph.get("signature") == p0.get("signature")),
                             reached=bool(np.asarray(auxh["reached"]))))
        per_out = {}
        for q, name in enumerate(J_NAMES):
            Eq = np.array([r["E"][q] for r in rows])
            hs = np.array(H_DEG)
            in_win = np.array([r["same_window"] and r["reached"] for r in rows])
            small = hs <= 1e-6
            noise = float(np.max(Eq[small])) if np.any(small) else 0.0
            usable = in_win & (Eq >= NOISE_FLOOR_FACTOR * noise) & (Eq > 0)
            idx = np.nonzero(usable)[0]
            run = []
            if idx.size:
                run = [idx[0]]
                for k in idx[1:]:
                    if k == run[-1] + 1:
                        run.append(k)
                    else:
                        break
            if len(run) >= 2:
                lh, lE = np.log10(hs[run]), np.log10(Eq[run])
                slope = float(np.polyfit(lh, lE, 1)[0])
                decades = float(lh.max() - lh.min())
            else:
                slope, decades = None, 0.0
            local = [float((math.log10(Eq[k]) - math.log10(Eq[k + 1])) / (math.log10(hs[k]) - math.log10(hs[k + 1])))
                     if Eq[k] > 0 and Eq[k + 1] > 0 else None for k in range(len(hs) - 1)]
            per_out[name] = dict(E=Eq.tolist(), in_window=in_win.tolist(), noise_floor=noise,
                                 fit_h_deg=[float(hs[k]) for k in run], fitted_slope=slope,
                                 fit_decades=decades, local_slopes=local,
                                 passes=bool(slope is not None and slope >= 1.8 and decades >= 2.0 - 1e-12),
                                 J=float(J0[q]), dJ_dthetaB_per_rad=float(dJ0[q]))
        taylor.append(dict(theta_B_deg=t_deg, window=dict((k, v) for k, v in w.items()),
                           pattern_n_active=p0["n_active"], pattern_n_special=p0["n_special"],
                           pattern_n_raw=p0["n_raw"], h_rows=rows, outputs=per_out))
    timings["taylor_eval_s"] = time.time() - t0

    t0 = time.time()
    host_parity = []
    for tp in taylor:
        tb = math.radians(tp["theta_B_deg"])
        kh = build_kernel(RT, RD, tb, GAMMA, N_KERNEL, starting_line_method="kliegel_levine",
                          max_rrcs=500, dtheta_limit=DTL, Ru=RU)
        bd = kh.rrcs[-1]
        wall, axis = bd[0], bd[-1]
        host_parity.append(dict(theta_B_deg=tp["theta_B_deg"], host_reached_wall=bool(kh.reached_wall),
                                host_rows_incl_start=len(kh.rrcs), jax_active_rows=tp["pattern_n_active"],
                                host_M_B=float(wall.M), jax_M_B=tp["outputs"]["J2_M_B"]["J"],
                                host_x_axis=float(axis.x), jax_x_axis=tp["outputs"]["J3_x_D_axis"]["J"],
                                abs_diff_M_B=abs(float(wall.M) - tp["outputs"]["J2_M_B"]["J"]),
                                abs_diff_x_axis=abs(float(axis.x) - tp["outputs"]["J3_x_D_axis"]["J"])))
    timings["host_parity_s"] = time.time() - t0

    taylor_pass = all(tp["outputs"][nm]["passes"] for tp in taylor for nm in J_NAMES)
    events_pass = all(jump_report[nm]["n_jumps_without_pattern_change"] == 0 for nm in J_NAMES)
    distinct = len({tp["window"]["signature"] for tp in taylor}) == 3
    status = "PASS" if (taylor_pass and events_pass and distinct) else "FAIL"

    plt = style()
    figs = {}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.1), gridspec_kw=dict(width_ratios=[1.35, 1.0]))
    J1 = J[:, 0]
    J1ref = float(np.mean(J1[valid]))
    rel = 1e6 * (J1 / J1ref - 1.0)
    for k in np.nonzero(pattern_change)[0]:
        xb = 0.5 * (theta_deg[k] + theta_deg[k + 1])
        ax1.axvline(xb, color=MUTED, lw=0.45, alpha=0.55, zorder=0)
    jr = jump_report["J1_mdot_BD"]["_jumps"]
    ax1.plot(theta_deg[valid], rel[valid], ".", ms=2.2, color=INK, label=r"$J_1$ (BD mass flow)")
    if np.any(jr):
        kk = np.nonzero(jr)[0]
        ax1.plot(0.5 * (theta_deg[kk] + theta_deg[kk + 1]), 0.5 * (rel[kk] + rel[kk + 1]), "o", mfc="none",
                 mec=ACCENT, ms=4.5, mew=0.8, label="detected jump")
    if np.any(~valid):
        ax1.plot(theta_deg[~valid], np.zeros(np.sum(~valid)), "x", color=ACCENT, ms=3, label="incomplete march")
    ax1.set_xlabel(r"$\theta_B$ (deg)")
    ax1.set_ylabel(r"$10^6\,(J_1/\bar J_1-1)$ (ppm)")
    ax1t = ax1.twinx()
    ax1t.plot(theta_deg[valid], 1e6 * dJ[valid, 0] / J1ref * (math.pi / 180.0), "-", lw=0.6, color=TEAL)
    ax1t.set_ylabel(r"AD $dJ_1/d\theta_B$ / $\bar J_1$ (ppm/deg)", color=TEAL)
    ax1t.tick_params(axis="y", colors=TEAL)
    ax1t.spines["right"].set_color(TEAL)
    ax1.plot([], [], "-", color=MUTED, lw=0.6, alpha=0.6, label="pattern change")
    ax1.plot([], [], "-", color=TEAL, lw=0.8, label="AD derivative (right)")
    ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), frameon=False, handlelength=1.5, ncol=2,
               borderaxespad=0.0)
    ax1.text(0.01, 0.98, "(a)", transform=ax1.transAxes, va="top", ha="left", fontsize=9)
    colors = [INK, TEAL, ACCENT]
    markers = ["o", "s", "^"]
    hs = np.array(H_DEG)
    for p, tp in enumerate(taylor):
        for q, nm in enumerate(J_NAMES):
            Eq = np.array(tp["outputs"][nm]["E"])
            inw = np.array(tp["outputs"][nm]["in_window"])
            pos = Eq > 0
            ax2.loglog(hs[pos & inw], Eq[pos & inw], "-", marker=markers[p], ms=3, lw=0.7, color=colors[q],
                       mfc=colors[q] if inw.all() else "none")
            if np.any(pos & ~inw):
                ax2.loglog(hs[pos & ~inw], Eq[pos & ~inw], marker=markers[p], ls="none", ms=3, mfc="none",
                           mec=colors[q], mew=0.6)
    href = np.array([1e-4, 1e-1])
    E_anchor = np.nanmax([taylor[0]["outputs"][nm]["E"][2] for nm in J_NAMES])
    ax2.loglog(href, E_anchor * (href / 1e-3) ** 2 * 3, "--", color=MUTED, lw=0.8)
    ax2.text(href[0] * 2.0, E_anchor * (href[0] * 2.0 / 1e-3) ** 2 * 30, "slope 2", color=MUTED, fontsize=7.5,
             ha="right")
    for q, nm in enumerate(["$J_1$", "$J_2$", "$J_3$"]):
        ax2.plot([], [], "-", color=colors[q], lw=0.8, label=nm)
    for p, tp in enumerate(taylor):
        ax2.plot([], [], ls="none", marker=markers[p], color=MUTED, ms=3,
                 label=rf"$\theta_B$={tp['theta_B_deg']:.3f}$^\circ$")
    ax2.set_xlabel(r"step $h$ (deg)")
    ax2.set_ylabel(r"$E(h)=|J(t+h)-J(t)-hJ'(t)|$")
    ax2.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), frameon=False, ncol=3, fontsize=6.5,
               handlelength=1.3, columnspacing=0.8, borderaxespad=0.0)
    ax2.text(0.02, 0.98, "(b)", transform=ax2.transAxes, va="top", ha="left", fontsize=9)
    fig.tight_layout()
    base = os.path.join(args.out, "e12_taylor_events")
    fig.savefig(base + ".pdf"); fig.savefig(base + ".png", dpi=200)
    plt.close(fig)
    figs["e12_taylor_events"] = base

    fig, axs = plt.subplots(2, 1, figsize=(6.5, 3.6), sharex=True)
    for ax, q, lab in [(axs[0], 1, r"$J_2=M_B$ (-)"), (axs[1], 2, r"$J_3=x_D/R_t$ (-)")]:
        for k in np.nonzero(pattern_change)[0]:
            ax.axvline(0.5 * (theta_deg[k] + theta_deg[k + 1]), color=MUTED, lw=0.45, alpha=0.55, zorder=0)
        ax.plot(theta_deg[valid], J[valid, q], ".", ms=2.0, color=INK)
        jj = jump_report[J_NAMES[q]]["_jumps"]
        if np.any(jj):
            kk = np.nonzero(jj)[0]
            ax.plot(0.5 * (theta_deg[kk] + theta_deg[kk + 1]), 0.5 * (J[kk, q] + J[kk + 1, q]), "o", mfc="none",
                    mec=ACCENT, ms=4.0, mew=0.8)
        ax.set_ylabel(lab)
        axt = ax.twinx()
        axt.plot(theta_deg[valid], dJ[valid, q] * math.pi / 180.0, "-", lw=0.6, color=TEAL)
        axt.set_ylabel(r"AD $dJ/d\theta_B$ (1/deg)", color=TEAL)
        axt.tick_params(axis="y", colors=TEAL)
    axs[1].set_xlabel(r"$\theta_B$ (deg)")
    fig.tight_layout()
    base2 = os.path.join(args.out, "e12_sweep_J2_J3")
    fig.savefig(base2 + ".pdf"); fig.savefig(base2 + ".png", dpi=200)
    plt.close(fig)
    figs["e12_sweep_J2_J3"] = base2

    np.savez_compressed(os.path.join(args.out, "e12_sweep.npz"), theta_B_deg=theta_deg, J=J, dJ_dthetaB_per_rad=dJ,
                        reached_wall=reached, bd_axis_idx=axis_idx,
                        n_active=np.array([p["n_active"] if p["n_active"] is not None else -1 for p in patterns]),
                        n_special=np.array([p["n_special"] if p["n_special"] is not None else -1 for p in patterns]),
                        pattern_change=pattern_change)

    wall = time.time() - t_start
    script = os.path.abspath(__file__)
    srcs = raosim_source_hashes()
    figure_hashes = {}
    for nm, b in figs.items():
        figure_hashes[nm] = {"pdf": os.path.relpath(b + ".pdf", ROOT), "pdf_sha256": sha256_file(b + ".pdf"),
                             "png": os.path.relpath(b + ".png", ROOT), "png_sha256": sha256_file(b + ".png")}
    npz = os.path.join(args.out, "e12_sweep.npz")
    for nm in J_NAMES:
        for key in ["_r_val", "_r_der", "_jumps"]:
            jump_report[nm].pop(key)
    result = dict(
        id="E12", status=status, evidence_class="verification",
        title="Taylor and event test of the JAX throat kernel derivative with respect to theta_B",
        command=" ".join([".venv-jax/bin/python", os.path.relpath(script, ROOT)] + list(argv if argv is not None else sys.argv[1:])),
        git_head=git_head(), worktree_note="working tree dirty at run time; script untracked",
        script=os.path.relpath(script, ROOT), script_sha256=sha256_file(script),
        raosim_sources_sha256=srcs,
        inputs=dict(note="no file inputs; start line generated by lrekit.components.nozzle.solvers.nasa.build_kernel (KL, Ru=1.5)",
                    start_line_sha256=sha256_arrays(*sl), start_line_theta_B_independent=bool(sl_identical),
                    t19_inputs_used=False),
        configuration=dict(Rt=RT, Rd_over_Rt=RD, gamma=GAMMA, n_kernel=N_KERNEL, Ru_over_Rt=RU,
                           starting_line_method="kliegel_levine", dtheta_limit_deg=0.25, dtheta_limit_rad=DTL,
                           n_corr=40, n_rows_static=n_rows,
                           n_rows_rule="theta_b_solve._n_rows_bound(radians(28.5), dtl) + %d" % ROW_MARGIN,
                           sweep_theta_B_deg=[TB_LO_DEG, TB_HI_DEG], sweep_points=args.n_points,
                           vmap_chunk=args.chunk, h_deg=H_DEG, taylor_targets_deg=TAYLOR_TARGETS_DEG,
                           J1="assembly._polyline_mass_flux(bd, gamma, _CE_M_FLOOR): mdot/(rho0 sqrt(R T0) Rt^2)",
                           J2="bd.M[0] (Mach at B)", J3="bd.x[bd_axis_idx] (x/Rt of BD axis point)",
                           derivative="jax.jvp (forward mode) w.r.t. theta_B in radians",
                           jump_rule=("pattern-blind: interval flagged if trapezoid residual |dJ - dt*(J'_i+J'_i+1)/2| or "
                                      "|J'_i+1 - J'_i| exceeds median + %g*1.4826*MAD over all complete intervals" % JUMP_MAD_K),
                           taylor_fit_rule=("fit log10 E vs log10 h over the leading contiguous run of h with t+h in the same "
                                            "pattern window and E >= %g x noise floor (max E at h <= 1e-6 deg)" % NOISE_FLOOR_FACTOR),
                           window_definition="maximal run of sweep points with identical (active row count, raw/special flag string)"),
        timings=timings, wall_time_s=wall,
        metrics=dict(n_incomplete_marches=n_incomplete, n_windows=len(windows),
                     n_pattern_changes=int(pattern_change.sum()), n_rowcount_changes=int(rowcount_change.sum()),
                     J1_mean=J1ref, J1_rel_span=float((J1[valid].max() - J1[valid].min()) / J1ref),
                     taylor_all_pass=taylor_pass, events_all_explained=events_pass,
                     taylor_windows_distinct=distinct,
                     taylor_summary=[dict(theta_B_deg=tp["theta_B_deg"],
                                          **{nm: dict(slope=tp["outputs"][nm]["fitted_slope"],
                                                      decades=tp["outputs"][nm]["fit_decades"],
                                                      passes=tp["outputs"][nm]["passes"]) for nm in J_NAMES})
                                     for tp in taylor],
                     jumps={nm: dict((k, v) for k, v in jump_report[nm].items()
                                     if k in ("n_jumps", "n_value_jumps", "n_derivative_jumps",
                                              "n_jumps_without_pattern_change", "n_pattern_changes_without_detected_jump"))
                            for nm in J_NAMES}),
        windows=windows, jump_report=jump_report, taylor=taylor, host_parity_at_taylor_points=host_parity,
        hypotheses=dict(
            taylor_second_order_inside_window=("supported" if taylor_pass else "not supported"),
            jumps_coincide_with_pattern_changes=("supported" if events_pass else "not supported")),
        figures=figure_hashes,
        data_files={os.path.relpath(npz, ROOT): sha256_file(npz)},
        caption_facts=dict(
            configuration=("JAX march_kernel on the Nozzle-B throat arc (Rt 1, Rd 0.382 Rt, gamma 1.23, host KL start "
                           "line n=24, dtheta_limit 0.25 deg, n_rows %d); theta_B swept 27.0-28.5 deg in %d points, "
                           "forward-mode AD derivative; Taylor remainders at three points in three different "
                           "raw/special pattern windows." % (n_rows, args.n_points)),
            evidence_class="verification",
            not_established=("Derivatives of the full forward net, the start line (build_start_line is "
                             "NotImplementedError), or the BVP solution; physical accuracy of the kernel; "
                             "behaviour for other dtheta_limit or n_kernel.")),
        not_established="Derivatives of the full forward net, start line (build_start_line is NotImplementedError) or BVP solution.",
        notes=[
            "Acceptance is evaluated exactly as pre-registered in configuration.jump_rule and taylor_fit_rule.",
            "Follow-up attribution of jumps that do not coincide with row-count changes: "
            "builds/moc_extensions_plan_2026-11-21/opus_b/e12/diagnostic_branch/result.json "
            "(tests/experiments/nozzle/moc_ext_e12_branch_switch_diagnostic.py, DIAGNOSTIC).",
        ],
    )
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(result, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(dict(status=status, metrics=result["metrics"], timings=timings, wall=wall), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
