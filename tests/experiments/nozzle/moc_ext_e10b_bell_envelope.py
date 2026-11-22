"""E10b: bell wall states past the fold (DIAGNOSTIC) and a planar simple-wave cross-check
(MODEL-APPROXIMATION) for the E10 envelope (H2) and envelope-consistent deflection (H3 ii).

Answer A5 (opus_a/answers_round1.json). E10's original status (INCOMPLETE) is unchanged.
E10b-1: the t01 nets (on_invalid='refine') are continued past folds with the fold audit made
non-terminating inside this script (moc_bounded._build_row called with orientation 0); reversed
cells are recorded post hoc with the audit's own area test. Bell wall nodes are accepted only if
x_w < x_rev,min - h and theta + mu < 90 deg (outside the domain of influence of the first reversed
or invalid cell).
E10b-2: planar simple-wave wall states nu_w = nu_N - (theta_N - theta_w) (C+ invariant theta - nu
uniform across a compression; anderson Eqs. 4.39-4.45, PDF 186-187; courant1944 simple waves as
cited by the planner), straight C- lines, cusp vs wall spacing.

    .venv-jax/bin/python tests/experiments/nozzle/moc_ext_e10b_bell_envelope.py
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
sys.path.insert(0, str(ROOT / "tests/experiments/nozzle"))
OUT = ROOT / "builds/moc_extensions_plan_2026-11-21/opus_b/e10b"

import numpy as np  # noqa: E402

import moc_ext_e10_top_fold_envelope as e10  # noqa: E402
import lrekit.components.nozzle.solvers.bounded as mb  # noqa: E402
from lrekit.core.gas_dynamics import mach_from_prandtl_meyer, prandtl_meyer  # noqa: E402
from lrekit.components.nozzle.solvers.moc import solve_flowfield  # noqa: E402

INK, TEAL, MUTED, ACCENT = "#19333C", "#176D76", "#51616B", "#C26A2E"
RT = e10.RT
SW_COUNTS = (50, 100, 200, 400, 800, 1600, 3200, 6400, 12800)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def continued_march(wall, n, dth):
    orig = mb._build_row
    invalid = []

    def wrapped(*a, **k):
        k = dict(k)
        k["orientation"] = 0.0
        out = orig(*a, **k)
        if out is None:
            invalid.append({"row_index": k.get("row_index"), "wall_x": float(a[1].x)})
        return out
    mb._build_row = wrapped
    try:
        t0 = time.time()
        res = solve_flowfield(RT, e10.EPS, e10.GAMMA, wall, n_char=n, dtheta_limit_deg=dth,
                              mass_tol=e10.MASS_TOL, on_invalid="refine")
        wt = time.time() - t0
    finally:
        mb._build_row = orig
    return res["net"], invalid, wt


def oriented_cells(prev, new, special, orient):
    i_last_prev = len(prev) - 1
    out = []
    for i in range(1, len(new) - 1):
        ii = min(i if special else i + 1, i_last_prev)
        if ii < 1:
            continue
        a = mb._signed_cell_area(prev[ii - 1], prev[ii], new[i], new[i - 1])
        e = math.hypot(prev[ii].x - prev[ii - 1].x, prev[ii].r - prev[ii - 1].r) * math.hypot(
            new[i].x - prev[ii].x, new[i].r - prev[ii].r)
        xs = (prev[ii - 1].x, prev[ii].x, new[i].x, new[i - 1].x)
        out.append({"i": i, "ii": ii, "area_norm": orient * a / max(e, 1e-300), "x_min": min(xs),
                    "x": new[i].x, "r": new[i].r})
    return out


def reversed_cells(net):
    a2 = [c["area_norm"] for c in oriented_cells(net.rows[1], net.rows[2], net.special[2], 1.0)]
    orient = math.copysign(1.0, float(np.median([a for a in a2 if a != 0.0])))
    rev = []
    for j in range(3, net.n_rows):
        for c in oriented_cells(net.rows[j - 1], net.rows[j], net.special[j], orient):
            if c["area_norm"] <= 1e-12:
                rev.append({"row": j, **c})
    return rev, orient


def envelope_from_states(W, wall):
    W = np.asarray(W)
    x, r, th, M = W.T
    s = np.tan(th - np.arcsin(1.0 / M))
    adj = []
    for k in range(len(x) - 1):
        ds = s[k] - s[k + 1]
        if abs(ds) < 1e-15:
            continue
        xs = (r[k + 1] - r[k] + s[k] * x[k] - s[k + 1] * x[k + 1]) / ds
        rs = r[k] + s[k] * (xs - x[k])
        inside = bool(xs > x[k + 1] and xs <= wall.x_end and 0.0 <= rs <= wall.r(min(xs, wall.x_end)))
        adj.append((k, float(xs), float(rs), inside))
    valid = [q for q in adj if q[3]]
    cusp = min(valid, key=lambda q: q[1]) if valid else None
    coal = []
    for k in range(1, len(x)):
        ds = s[:k] - s[k]
        with np.errstate(divide="ignore", invalid="ignore"):
            xs = (r[k] - r[:k] + s[:k] * x[:k] - s[k] * x[k]) / ds
        rs = r[:k] + s[:k] * (xs - x[:k])
        ok = np.isfinite(xs) & (xs > x[k]) & (xs <= wall.x_end) & (rs >= 0.0)
        if ok.any():
            rw = np.array([wall.r(v) if o else -1.0 for v, o in zip(xs, ok)])
            ok &= rs <= rw
        if ok.any():
            coal.append(k)
    dth = sum(max(th[k - 1] - th[k], 0.0) for k in coal)
    down = [q for q in adj if q[1] > x[q[0] + 1]]
    stats = {"n_downstream_pairs": len(down), "n_diverging_pairs_meeting_upstream": len(adj) - len(down),
             "downstream_pair_x_range_m": [min(q[1] for q in down), max(q[1] for q in down)] if down else None,
             "downstream_pair_r_range_m": [min(q[2] for q in down), max(q[2] for q in down)] if down else None,
             "n_downstream_pairs_with_r_ge_0": sum(1 for q in down if q[2] >= 0.0),
             "axis_hit_x_range_m": [float(np.min(x - r / s)), float(np.max(x - r / s))]}
    return {"cusp": cusp, "n_pairs": len(adj), "n_inside": len(valid), "coalescing": coal,
            "deflection_rad": float(dth), "stats": stats}


def simple_wave_states(wall, x_N, M_N, theta_N, n_w, g):
    xs = np.linspace(x_N, wall.x_end, n_w)
    nu_N = prandtl_meyer(M_N, g)
    W = []
    for x in xs:
        th = float(wall.theta(x))
        nu = nu_N - (theta_N - th)
        if nu <= 0.0:
            break
        W.append((float(x), float(wall.r(x)), th, mach_from_prandtl_meyer(nu, g, tol=1e-13, max_iter=200)))
    return np.array(W)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    wall, contour = e10.bezier_wall()
    x_N = float(contour["N"][0])
    levels = []
    for n, dth in e10.LEVELS:
        _, net0, crossing0, wt0, _ = e10.run_level(wall, n, dth, "refine")
        fold = np.array(net0.fold_location)
        near = min(net0.rows[-1], key=lambda p: math.hypot(p.x - fold[0], p.r - fold[1]))
        h_loc_fold = e10.dist_point_polyline(np.array([near.x, near.r]), e10._poly(net0.rows[-2]))
        (M1, th1), _ = e10.interp_on_row(net0.rows[-1], fold)
        N_state = net0.rows[-1][0]

        net, invalid, wt = continued_march(wall, n, dth)
        rev, orient = reversed_cells(net)
        cand = [{"x": c["x_min"], "kind": "reversed", "row": c["row"], "ii": c["ii"]} for c in rev]
        cand += [{"x": v["wall_x"], "kind": "invalid", "row": v["row_index"], "ii": None} for v in invalid]
        first = min(cand, key=lambda c: c["x"]) if cand else None
        h_rev = None
        if first is not None and first["kind"] == "reversed" and first["row"] >= 3:
            p = net.rows[first["row"] - 1][first["ii"]]
            h_rev = e10.dist_point_polyline(np.array([p.x, p.r]), e10._poly(net.rows[first["row"] - 2]))
        x_cut = (first["x"] - (h_rev if h_rev is not None else h_loc_fold)) if first else math.inf
        bell = [row[0] for row in net.rows[1:] if row[0].x > x_N + 1e-12]
        acc = [p for p in bell if p.x < x_cut and math.degrees(p.theta + math.asin(1.0 / p.M)) < 90.0]
        W = [(N_state.x, N_state.r, N_state.theta, N_state.M)] + [(p.x, p.r, p.theta, p.M) for p in acc]
        env = envelope_from_states(W, wall) if len(W) >= 2 else {"cusp": None, "n_pairs": 0, "n_inside": 0,
                                                                  "coalescing": [], "deflection_rad": 0.0,
                                                                  "stats": None}
        acc_store = W
        cusp = env["cusp"]
        dist = math.hypot(cusp[1] - fold[0], cusp[2] - fold[1]) if cusp else None
        sh = e10.weak_shock(M1, env["deflection_rad"], e10.GAMMA)
        levels.append({
            "n": n, "dtheta_deg": dth,
            "original_refine_fold_m": fold.tolist(), "M1_at_fold": M1, "h_loc_at_fold_m": h_loc_fold,
            "N_state": {"x": N_state.x, "r": N_state.r, "theta_deg": math.degrees(N_state.theta), "M": N_state.M},
            "continued_march": {"stop_reason": net.stop_reason, "complete": bool(net.complete), "n_rows": net.n_rows,
                                "max_abs_mdot_rel_err": float(max(abs(v) for v in net.mdot_rel_err)),
                                "wall_time_s": wt, "n_reversed_cells": len(rev), "n_invalid_rows": len(invalid),
                                "n_bell_wall_nodes": len(bell), "orientation": orient},
            "first_reversed_or_invalid": ({**first, "h_row_m": h_rev} if first else None),
            "x_cut_m": x_cut,
            "accepted_bell_wall_nodes": {"count": len(acc),
                                         "x_range_m": [acc[0].x, acc[-1].x] if acc else None,
                                         "x_range_over_Rt": [acc[0].x / RT, acc[-1].x / RT] if acc else None,
                                         "theta_mu_max_deg": max(math.degrees(p.theta + math.asin(1 / p.M))
                                                                 for p in acc) if acc else None,
                                         "M_first_last": [acc[0].M, acc[-1].M] if acc else None,
                                         "theta_first_last_deg": [math.degrees(acc[0].theta),
                                                                  math.degrees(acc[-1].theta)] if acc else None},
            "H2": {"cusp_xr_m": [cusp[1], cusp[2]] if cusp else None, "distance_to_fold_m": dist,
                   "distance_over_h_loc": (dist / h_loc_fold) if dist is not None else None,
                   "within_one_h_loc": (dist is not None and dist <= h_loc_fold),
                   "n_pairs": env["n_pairs"], "n_pairs_inside": env["n_inside"], "pair_stats": env["stats"],
                   "includes_N_state": True},
            "H3ii": {"n_coalescing_wall_C-": len(env["coalescing"]),
                     "deflection_deg": math.degrees(env["deflection_rad"]), "weak_shock": sh,
                     "p02_p01_ge_0.99": bool(sh.get("attached") and sh["p02_p01"] >= 0.99)},
        })
        print(n, net.stop_reason, len(rev), first, len(acc), cusp, dist, flush=True)

    fin = levels[-1]
    M_N, th_N = fin["N_state"]["M"], math.radians(fin["N_state"]["theta_deg"])
    sw = []
    sw_store = None
    for nw in SW_COUNTS:
        W = simple_wave_states(wall, x_N, M_N, th_N, nw, e10.GAMMA)
        if nw == 400:
            sw_store = W
        env = envelope_from_states(W, wall)
        c = env["cusp"]
        sw.append({"n_wall": nw, "spacing_m": (wall.x_end - x_N) / (nw - 1),
                   "cusp_xr_m": [c[1], c[2]] if c else None, "cusp_k": c[0] if c else None,
                   "cusp_wall_x_m": float(W[c[0], 0]) if c else None,
                   "deflection_envelope_deg": math.degrees(env["deflection_rad"]),
                   "n_coalescing": len(env["coalescing"]), "pair_stats": env["stats"]})
    fold_f = np.array(fin["original_refine_fold_m"])
    last = sw[-1]
    sw_summary = {
        "cusp_limit_estimate_xr_m": last["cusp_xr_m"],
        "last_two_change_m": (math.hypot(sw[-1]["cusp_xr_m"][0] - sw[-2]["cusp_xr_m"][0],
                                         sw[-1]["cusp_xr_m"][1] - sw[-2]["cusp_xr_m"][1])
                              if sw[-1]["cusp_xr_m"] and sw[-2]["cusp_xr_m"] else None),
        "distance_to_finest_fold_m": (math.hypot(last["cusp_xr_m"][0] - fold_f[0], last["cusp_xr_m"][1] - fold_f[1])
                                      if last["cusp_xr_m"] else None),
        "M_N_used": M_N, "theta_N_deg": math.degrees(th_N),
        "deflection_weak_shock_at_M1_finest": e10.weak_shock(fin["M1_at_fold"],
                                                             math.radians(last["deflection_envelope_deg"]), e10.GAMMA),
    }
    figs = figure(wall, contour, levels, sw, acc_store, sw_store)
    any_cusp = [L for L in levels if L["H2"]["cusp_xr_m"]]
    result = {
        "id": "E10b", "status": "PASS-DIAGNOSTIC",
        "labels": ["DIAGNOSTIC (E10b-1)", "MODEL-APPROXIMATION, planar (E10b-2)", "E10 original status unchanged"],
        "evidence_class": "neither (diagnostic and model-approximation)",
        "from_answer": "A5 (and A6) in opus_a/answers_round1.json",
        "configuration": {
            "case": "E10/t01 current-chart bell, the four t01 levels, on_invalid='refine' (planner-confirmed primary)",
            "continued_march": "solve_flowfield with moc_bounded._build_row wrapped in this script to pass "
                               "orientation 0 (fold audit non-terminating); reversed cells found post hoc with the "
                               "audit's normalised signed-area test and the row-2 orientation; invalid rows "
                               "(unit-process failure) bounded below by their wall x",
            "acceptance": "bell wall node accepted if x_w < x_rev,min - h and theta_w + mu_w < 90 deg; "
                          "x_rev,min = smallest corner x of any reversed cell or wall x of any invalid row; "
                          "h = distance from the reversed cell's previous-row corner to the row before",
            "H2": "straight C- (slope tan(theta_w - mu_w)) from adjacent nodes among N and the accepted bell nodes; "
                  "earliest intersection downstream of both origins with 0 <= r <= r_wall and x <= x_E",
            "H3ii": "sum of positive wall-angle decrements of accepted wall C- whose straight extension crosses an "
                    "earlier one inside the nozzle; weak oblique shock at the level's fold Mach (anderson Eqs. 4.7-"
                    "4.12, 4.17; 3.62-3.63)",
            "simple_wave": "nu_w = nu(M_N) - (theta_N - theta_w), M_N and theta_N from the finest refine net's N; "
                           "wall sampled uniformly in x on the bell with n_wall points; anderson Eqs. 4.39-4.45 "
                           "(PDF 186-187); courant1944 (as cited by the planner; page not checked here). Ignores the "
                           "axisymmetric source term and the non-uniform kernel invariants.",
        },
        "command": ".venv-jax/bin/python tests/experiments/nozzle/moc_ext_e10b_bell_envelope.py",
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                   text=True).stdout.strip(),
        "hashes": {"script": {str(Path(__file__).resolve().relative_to(ROOT)): sha256(Path(__file__)),
                              "tests/experiments/nozzle/moc_ext_e10_top_fold_envelope.py":
                                  sha256(ROOT / "tests/experiments/nozzle/moc_ext_e10_top_fold_envelope.py")},
                   "raosim_imported": e10.raosim_hashes(),
                   "inputs": {str(e10.T01.relative_to(ROOT)): sha256(e10.T01)}},
        "metrics": {"levels": levels, "simple_wave": sw, "simple_wave_summary": sw_summary},
        "hypotheses": {
            "H2_E10b1": ("supported at " + ", ".join(f"n {L['n']}" for L in levels if L["H2"]["within_one_h_loc"])
                         if any(L["H2"]["within_one_h_loc"] for L in levels) else
                         ("not supported (cusp found but farther than one h_loc from the fold at every level)"
                          if any_cusp else
                          "not supported: straight C- from the accepted bell wall nodes have no intersection inside "
                          "the nozzle at any level (adjacent pairs meet at r < 0, beyond the axis), while the net "
                          "folds at x about 2.5 Rt; the planar straight-C- estimate does not locate the fold")),
            "H3ii_E10b1_verdict": ("vacuous: no accepted wall C- crosses an earlier one inside the nozzle, so the "
                                   "envelope-consistent deflection is 0 at every level"
                                   if all(L["H3ii"]["deflection_deg"] == 0.0 for L in levels) else "see H3ii_E10b1"),
            "H3ii_E10b1": [{"n": L["n"], "deflection_deg": L["H3ii"]["deflection_deg"],
                            "p02_p01": L["H3ii"]["weak_shock"].get("p02_p01")} for L in levels],
            "H2_E10b2_simple_wave": sw_summary,
        },
        "findings": [],
        "figures": [{"paths": figs, "caption_facts": {
            "configuration": "Current-chart TOP bell (E10/t01 case). Left: wall, axis, exit plane (dashed), net fold "
                             "points (on_invalid 'refine'), and a subset of straight C- lines from the accepted net "
                             "wall nodes of the continued march, n 144 (DIAGNOSTIC), and from planar simple-wave wall "
                             "states (MODEL-APPROXIMATION); neither family meets inside the nozzle. Right: wall Mach "
                             "along the bell for both.",
            "evidence_class": "neither (DIAGNOSTIC / MODEL-APPROXIMATION)",
            "not_established": "The physical internal-shock position; axisymmetric envelope; validity of the "
                               "continued march beyond the first reversed cell."}}],
        "wall_time_s": None,
        "notes": ["E10 result.json is unchanged except for the planner-requested annotation block.",
                  "The earlier B2 proposal nu_w = nu_N + (theta_N - theta_w) was the expansion form; the "
                  "compression form used here follows answer A5."],
    }
    for L in levels:
        result["findings"].append(
            f"n {L['n']}: continued march {L['continued_march']['stop_reason']} with "
            f"{L['continued_march']['n_reversed_cells']} reversed cells; first at x = "
            f"{L['first_reversed_or_invalid']['x'] if L['first_reversed_or_invalid'] else float('nan'):.5f} m; "
            f"{L['accepted_bell_wall_nodes']['count']} accepted bell wall nodes; cusp "
            f"{L['H2']['cusp_xr_m']} vs fold {L['original_refine_fold_m']} "
            f"(distance/h_loc {L['H2']['distance_over_h_loc']}); H3(ii) deflection "
            f"{L['H3ii']['deflection_deg']:.3f} deg, p02/p01 {L['H3ii']['weak_shock'].get('p02_p01')}.")
    result["findings"].append(
        "In the net the wall Mach rises along the accepted bell span (n 144: M "
        f"{fin['accepted_bell_wall_nodes']['M_first_last'][0]:.3f} -> {fin['accepted_bell_wall_nodes']['M_first_last'][1]:.3f} "
        f"while theta falls {fin['accepted_bell_wall_nodes']['theta_first_last_deg'][0]:.2f} -> "
        f"{fin['accepted_bell_wall_nodes']['theta_first_last_deg'][1]:.2f} deg), so straight wall C- mostly diverge "
        f"({fin['H2']['pair_stats']['n_diverging_pairs_meeting_upstream']} of {fin['H2']['n_pairs']} adjacent pairs meet "
        "upstream). The planar simple wave instead compresses (M falls) and its straight C- converge, but only below "
        "the axis and downstream of the exit. Neither straight-C- construction reproduces the fold at x about 2.5 Rt, "
        "r about 0.95 Rt, which the net forms through curved C- in the non-uniform axisymmetric field.")
    result["findings"].append(
        f"Simple wave: cusp {sw_summary['cusp_limit_estimate_xr_m']} at {SW_COUNTS[-1]} wall points (last change "
        f"{sw_summary['last_two_change_m']} m), {sw_summary['distance_to_finest_fold_m']} m from the finest fold.")
    result["wall_time_s"] = time.time() - t0
    for _f in result["figures"]:
        _f.update({"script": str(Path(__file__).resolve().relative_to(ROOT)), "command": result["command"],
                   "git_head": result["git_head"], "status": result["status"],
                   "wall_time_s": result["wall_time_s"], "hashes": "see result.hashes (script, raosim, inputs)",
                   "file_sha256": {q: sha256(ROOT / q) for q in _f["paths"]}})
    (OUT / "result.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"hyp": result["hypotheses"], "findings": result["findings"]}, indent=1, default=float))


def figure(wall, contour, levels, sw, acc_states, sw_states):
    plt = e10._plt()
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.9), constrained_layout=True,
                            gridspec_kw={"width_ratios": [1.7, 1.0]})
    ax = axs[0]
    xa = np.array(contour["x"]) / RT
    ya = np.array(contour["y"]) / RT
    m = xa >= 0
    ax.plot(xa[m], ya[m], color=INK, lw=1.0)
    ax.axhline(0.0, color=MUTED, lw=0.6)
    ax.axvline(wall.x_end / RT, color=MUTED, lw=0.6, ls="--")
    xmax = 13.0

    def lines(W, col, ls, lab, every):
        W = np.asarray(W)
        idx = list(range(0, len(W), max(1, len(W) // every))) + [len(W) - 1]
        first = True
        for k in sorted(set(idx)):
            x0, r0, th, M = W[k]
            sl = math.tan(th - math.asin(1.0 / M))
            xx = np.array([x0, xmax * RT])
            ax.plot(xx / RT, (r0 + sl * (xx - x0)) / RT, color=col, lw=0.5, ls=ls, label=lab if first else None)
            first = False
    lines(acc_states, TEAL, "-", "straight C- from accepted net wall nodes (n 144)", 8)
    lines(sw_states, ACCENT, "--", "straight C-, planar simple wave", 8)
    mk = ["o", "s", "^", "D"]
    for L, m_ in zip(levels, mk):
        f = L["original_refine_fold_m"]
        ax.plot(f[0] / RT, f[1] / RT, m_, ms=4, mfc="white", mec=INK, mew=0.8, label=f"fold n {L['n']}")
    N = contour["N"]
    ax.plot(N[0] / RT, N[1] / RT, "P", ms=5, color=TEAL)
    ax.set_xlim(-0.1, xmax)
    ax.set_ylim(-1.0, math.sqrt(e10.EPS) * 1.05)
    ax.set_xlabel("$x/R_t$ (-)")
    ax.set_ylabel("$r/R_t$ (-)")
    ax.legend(frameon=False, fontsize=5.8, loc="lower left")
    ax2 = axs[1]
    A = np.asarray(acc_states)
    S = np.asarray(sw_states)
    ax2.plot(A[:, 0] / RT, A[:, 3], "-", color=TEAL, lw=1.1, label="net wall M (accepted, n 144)")
    ax2.plot(S[:, 0] / RT, S[:, 3], "--", color=ACCENT, lw=1.1, label="simple-wave wall M")
    ax2.axvline(levels[-1]["original_refine_fold_m"][0] / RT, color=MUTED, lw=0.6, ls=":")
    ax2.set_xlabel("wall $x/R_t$ (-)")
    ax2.set_ylabel("wall Mach number (-)")
    ax2.legend(frameon=False, fontsize=6.3)
    for a_ in axs:
        for s_ in ("top", "right"):
            a_.spines[s_].set_visible(False)
    paths = []
    for ext in ("pdf", "png"):
        p = OUT / f"e10b_bell_envelope.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(str(p.relative_to(ROOT)))
    plt.close(fig)
    return paths


if __name__ == "__main__":
    main()
