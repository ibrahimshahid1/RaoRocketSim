"""E12 follow-up (DIAGNOSTIC): attribute the E12 within-window jumps of J1/J3 to
branch switches of the NASA dtheta-form unit processes on the final (clamped)
BD row.  Branch conditions are recomputed in NumPy from the march output
(local copy of the conditions in raosim/jax/moc_kernel.py interior_point_nasa;
library code is not modified).

.venv-jax/bin/python tests/experiments/nozzle/moc_ext_e12_branch_switch_diagnostic.py
"""
from __future__ import annotations

from lrekit.paths import repository_root as _repository_root

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

ROOT = str(_repository_root())
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import lrekit.numerics.jax  # noqa: E402,F401
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from lrekit.numerics.jax.moc_kernel import march_kernel  # noqa: E402
from lrekit.numerics.jax.theta_b_solve import _n_rows_bound, start_line_from_kernel  # noqa: E402
from lrekit.components.nozzle.solvers.nasa import build_kernel  # noqa: E402

sys.path.insert(0, os.path.join(ROOT, "tests", "experiments", "nozzle"))
import moc_ext_e12_jax_kernel_taylor as e12  # noqa: E402

E12_DIR = os.path.join(ROOT, "builds", "moc_extensions_plan_2026-11-21", "opus_b", "e12")
OUT_DEFAULT = os.path.join(E12_DIR, "diagnostic_branch")
INK, TEAL, MUTED, ACCENT = e12.INK, e12.TEAL, e12.MUTED, e12.ACCENT


def mm(M):
    return np.sqrt(np.maximum(M * M - 1.0, 1e-30))


def mu_of(M):
    return np.arcsin(np.clip(1.0 / np.maximum(M, 1.000001), 0.0, 1.0))


def coef(kind, M, th, r):
    bad = (r == 0.0) | (np.abs(th) < 1e-9)
    t = np.tan(np.where(bad, 1.0, th))
    m = mm(M)
    den = {"B": r * (m / t - 1.0), "b": r * (m / t + 1.0), "R": r * (m + 1.0 / t), "Rs": r * (m - 1.0 / t)}[kind]
    bad = bad | (np.abs(den) < 1e-12)
    return np.where(bad, 0.0, 1.0 / np.where(bad, 1.0, den))


def branch_flags(bd, prev, axis_idx):
    """Per-interior-node flags of the final special row (pairing shift 0)."""
    flags = []
    for i in range(1, axis_idx):
        x2, r2, t2, M2 = bd["x"][i - 1], bd["r"][i - 1], bd["theta"][i - 1], max(bd["M"][i - 1], 1.000001)
        ii = min(i, len(prev["x"]) - 1)
        r1, t1, M1 = prev["r"][ii], prev["theta"][ii], max(prev["M"][ii], 1.000001)
        if r1 == 0.0:
            jo = max(ii - 1, 0)
            r1b, t1b, M1b = prev["r"][jo], prev["theta"][jo], max(prev["M"][jo], 1.000001)
        else:
            r1b, t1b, M1b = r1, t1, M1
        s1 = math.tan(t1 + mu_of(M1))
        s2 = math.tan(t2 - mu_of(M2))
        fT2 = abs(coef("b", M2, t2, r2)) <= abs(coef("Rs", M2, t2, r2))
        fT1 = abs(coef("B", M1b, t1b, r1b)) <= abs(coef("R", M1b, t1b, r1b))
        fr3 = abs(s2) <= abs(s1)
        fclamp = bd["theta"][i] == 0.0
        fbad = abs(t2) < 1e-9
        frzero = bd["r"][i] == 0.0
        flags.append((fT2, fT1, fr3, fclamp, fbad, frzero))
    return np.array(flags, dtype=bool)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunk", type=int, default=43)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()

    res = json.load(open(os.path.join(E12_DIR, "result.json")))
    npz_path = os.path.join(E12_DIR, "e12_sweep.npz")
    sw = np.load(npz_path)
    theta_deg = sw["theta_B_deg"]
    n = theta_deg.size
    theta_rad = np.radians(theta_deg)
    dtl = e12.DTL

    k_hi = build_kernel(e12.RT, e12.RD, math.radians(e12.TB_HI_DEG), e12.GAMMA, e12.N_KERNEL,
                        starting_line_method="kliegel_levine", max_rrcs=500, dtheta_limit=dtl, Ru=e12.RU)
    sl = start_line_from_kernel(k_hi)
    n_rows = _n_rows_bound(math.radians(e12.TB_HI_DEG), dtl) + e12.ROW_MARGIN
    assert n_rows == res["configuration"]["n_rows_static"]
    assert e12.sha256_arrays(*sl) == res["inputs"]["start_line_sha256"]

    def rows_of(tb):
        m = march_kernel(sl, tb, e12.RT, e12.RD, e12.GAMMA, n_rows=n_rows, n_corr=40, dtheta_limit=dtl)
        hit = m.wall_theta[1:] >= tb - 1e-9
        idx = jnp.argmax(hit) + 1
        prev = jax.tree.map(lambda a: a[idx - 1], m.rows)
        return m.bd, prev, m.bd_axis_idx, idx

    f = jax.jit(jax.vmap(rows_of))
    n_chunks = int(math.ceil(n / args.chunk))
    pad = n_chunks * args.chunk - n
    tb_pad = np.concatenate([theta_rad, np.full(pad, theta_rad[-1])])
    t0 = time.time()
    fc = f.lower(jnp.asarray(tb_pad[:args.chunk])).compile()
    t_compile = time.time() - t0
    t0 = time.time()
    out = []
    for c in range(n_chunks):
        bd, prev, ax, idx = fc(jnp.asarray(tb_pad[c * args.chunk:(c + 1) * args.chunk]))
        jax.block_until_ready(bd.x)
        for k in range(args.chunk):
            out.append((dict((nm, np.asarray(getattr(bd, nm))[k]) for nm in ("x", "r", "theta", "M")),
                        dict((nm, np.asarray(getattr(prev, nm))[k]) for nm in ("x", "r", "theta", "M")),
                        int(np.asarray(ax)[k]), int(np.asarray(idx)[k])))
    t_eval = time.time() - t0
    out = out[:n]

    J = sw["J"]
    J_check = np.array([[o[0]["M"][0], o[0]["x"][o[2]]] for o in out])
    max_dev = float(np.max(np.abs(J_check - J[:, 1:3])))

    flags = [branch_flags(o[0], o[1], o[2]) for o in out]
    names = ["T2_zform_vs_rform", "T1_zform_vs_rform", "r3_slope_select", "TH3_clamp_zero", "theta2_bad_guard",
             "r3_clamp_zero"]
    pattern_change = sw["pattern_change"]
    branch_change = np.zeros(n - 1, dtype=bool)
    branch_kind = []
    for i in range(n - 1):
        a, b = flags[i], flags[i + 1]
        if a.shape != b.shape:
            branch_change[i] = True
            branch_kind.append(["row_width"])
            continue
        diff = np.any(a != b, axis=0)
        branch_change[i] = bool(diff.any())
        branch_kind.append([names[k] for k in np.nonzero(diff)[0]])

    attribution = {}
    for q, nm in enumerate(e12.J_NAMES):
        jr = res["jump_report"][nm]
        jumps = np.zeros(n - 1, dtype=bool)
        for lo, hi in jr["jump_intervals_theta_deg"]:
            k = int(np.argmin(np.abs(theta_deg[:-1] - lo)))
            jumps[k] = True
        explained_pattern = jumps & pattern_change
        explained_branch = jumps & ~pattern_change & branch_change
        unexplained = jumps & ~pattern_change & ~branch_change
        events = np.nonzero(pattern_change | branch_change)[0]
        tail = []
        for k in np.nonzero(unexplained)[0]:
            prior = events[events < k]
            d_prev = float(theta_deg[k] - theta_deg[prior[-1] + 1]) if prior.size else float(theta_deg[k] - theta_deg[0])
            tail.append(dict(theta_deg=float(theta_deg[k]), deg_after_preceding_event_or_sweep_start=d_prev,
                             preceding="event" if prior.size else "sweep_start"))
        kinds = {}
        for k in np.nonzero(explained_branch)[0]:
            for kk in branch_kind[k]:
                kinds[kk] = kinds.get(kk, 0) + 1
        attribution[nm] = dict(
            n_jumps=int(jumps.sum()), explained_by_pattern_change=int(explained_pattern.sum()),
            explained_by_branch_switch_only=int(explained_branch.sum()),
            unexplained=int(unexplained.sum()),
            unexplained_theta_deg=[float(theta_deg[k]) for k in np.nonzero(unexplained)[0]],
            branch_switch_kinds=kinds,
            posthoc_unexplained_offsets=tail,
            posthoc_max_offset_after_event_deg=(max(t["deg_after_preceding_event_or_sweep_start"] for t in tail) if tail else None),
            branch_switch_intervals_without_detected_jump=int((branch_change & ~pattern_change & ~jumps).sum()))

    plt = e12.style()
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 2.8))
    dt = np.radians(np.diff(theta_deg))
    q = 2
    rv = np.abs(J[1:, q] - J[:-1, q] - dt * 0.5 * (sw["dJ_dthetaB_per_rad"][1:, q] + sw["dJ_dthetaB_per_rad"][:-1, q]))
    xm = 0.5 * (theta_deg[1:] + theta_deg[:-1])
    m0 = ~pattern_change & ~branch_change
    ax.semilogy(xm[m0], np.maximum(rv[m0], 1e-18), ".", ms=2.5, color=MUTED, label="no recorded change")
    m1 = ~pattern_change & branch_change
    ax.semilogy(xm[m1], np.maximum(rv[m1], 1e-18), "s", ms=3, mfc="none", mec=TEAL, mew=0.7,
                label="unit-process branch switch on BD row")
    ax.semilogy(xm[pattern_change], np.maximum(rv[pattern_change], 1e-18), "o", ms=3.5, mfc="none", mec=ACCENT,
                mew=0.8, label="row-count change")
    thr = res["jump_report"]["J3_x_D_axis"]["value_residual_threshold"]
    ax.axhline(thr, color=INK, lw=0.7, ls="--", label="E12 jump threshold")
    ax.set_xlabel(r"$\theta_B$ (deg)")
    ax.set_ylabel(r"$J_3$ trapezoid residual $|\Delta J-\Delta\theta\,\overline{J'}|$ ($R_t$)")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), frameon=False, ncol=4, fontsize=7, borderaxespad=0.0,
              handlelength=1.2, columnspacing=0.9)
    fig.tight_layout()
    base = os.path.join(args.out, "e12_branch_attribution_J3")
    fig.savefig(base + ".pdf"); fig.savefig(base + ".png", dpi=200)
    plt.close(fig)

    all_explained = all(attribution[nm]["unexplained"] == 0 for nm in e12.J_NAMES)
    script = os.path.abspath(__file__)
    result = dict(
        id="E12-diag-branch", status="PASS-DIAGNOSTIC" if all_explained else "FAIL-DIAGNOSTIC",
        evidence_class="verification (diagnostic attribution; does not change the E12 acceptance result)",
        command=" ".join([".venv-jax/bin/python", os.path.relpath(script, ROOT)] + list(argv if argv is not None else sys.argv[1:])),
        git_head=e12.git_head(), script=os.path.relpath(script, ROOT), script_sha256=e12.sha256_file(script),
        raosim_sources_sha256=e12.raosim_source_hashes(),
        inputs={os.path.relpath(npz_path, ROOT): e12.sha256_file(npz_path),
                os.path.relpath(os.path.join(E12_DIR, "result.json"), ROOT): e12.sha256_file(os.path.join(E12_DIR, "result.json")),
                "tests/experiments/nozzle/moc_ext_e12_jax_kernel_taylor.py": e12.sha256_file(e12.__file__)},
        configuration=dict(same_as="E12", branch_conditions=names,
                           note="conditions evaluated at the converged final-row nodes; final row is special (pairing shift 0)"),
        metrics=dict(recomputed_J2_J3_max_abs_dev_vs_E12=max_dev, n_branch_switch_intervals=int(branch_change.sum()),
                     n_branch_switch_intervals_inside_windows=int((branch_change & ~pattern_change).sum()),
                     attribution=attribution, all_jumps_explained=all_explained),
        timings=dict(jit_compile_s=t_compile, eval_s=t_eval), wall_time_s=time.time() - t_start,
        figures={"e12_branch_attribution_J3": {"pdf": os.path.relpath(base + ".pdf", ROOT), "pdf_sha256": e12.sha256_file(base + ".pdf"),
                                               "png": os.path.relpath(base + ".png", ROOT), "png_sha256": e12.sha256_file(base + ".png")}},
        caption_facts=dict(
            configuration=("Same JAX march and theta_B sweep as E12; trapezoid residual of J3 per sweep interval, "
                           "classified by whether the row count or a unit-process branch condition on the final BD row changes."),
            evidence_class="verification (diagnostic)",
            not_established="That the branch switches are the only source of non-smoothness for other outputs, gamma, n_kernel or dtheta_limit; any physical accuracy statement."),
    )
    with open(os.path.join(args.out, "result.json"), "w") as fh:
        json.dump(result, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(result["metrics"], indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
