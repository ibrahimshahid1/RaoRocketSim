"""E09b (DIAGNOSTIC): KL start-line construction review (K0a) for Nozzle-B, variants V0-V3 plus a swapped-basis pair.

Opt-in experiment script (moc_extensions_plan_2026-11-21). Local coefficient and normalisation variants only; the
production kliegel_levine is monkeypatched in-process (E09 copy) and restored. No BVP solve (D-K0-BVP not approved,
user decision D10).
Run: .venv-jax/bin/python -m tests.experiments.nozzle.moc_ext_e09b_startline_review
"""

from __future__ import annotations

import math

import numpy as np

from scripts import moc_ext_e00_common as C

RU = 1.5
GAMMA = 1.23
M_MIN = 1.05
N_LIST = (24, 48, 64)
DTHETA_FOR_N = {24: 0.25, 48: 0.0625, 64: 0.03125}
N_FIRST_ROWS = 40
THETA_B_DEG = 27.760614752769474
VARIANTS = {
    "V0": (+1.0, "mstar", "mstar"), "V1": (-1.0, "mstar", "mstar"), "V2": (+1.0, "true", "true"),
    "V3": (-1.0, "true", "true"), "V1a": (-1.0, "mstar", "true"), "V1b": (-1.0, "true", "mstar"),
}
DESC = {"V0": "repo sign; push on M*, M* as M (production)", "V1": "negated v3 z*y^3; push on M*, M* as M",
        "V2": "repo sign; push on M, M output", "V3": "negated; push on M, M output",
        "V1a": "negated; push on M*, M output (swapped output basis vs V1)",
        "V1b": "negated; push on M, M* output (swapped push basis vs V1)"}


def build_start_line(n, sign, push_basis, out_basis, Rt=1.0, Ru=RU, gamma=GAMMA):
    """Local copy of nasa_moc._make_throat_initial_line ('kliegel_levine' branch) with separate push/output bases."""
    import lrekit.components.nozzle.solvers.transonic as tk
    from lrekit.core.gas_dynamics import mach_angle, prandtl_meyer
    from lrekit.components.nozzle.solvers.moc import CharPoint
    from lrekit.components.nozzle.solvers.nasa import _push_throat_point_to_supersonic
    from tests.experiments.nozzle.moc_ext_e09_kl_variants import make_kl, mach_from_mstar
    kl_push = make_kl(sign, push_basis)
    kl_out = make_kl(sign, out_basis)
    kl_star = make_kl(sign, "mstar")
    nn = max(n - 1, 1)
    wall_first, info = [], []
    orig = tk.kliegel_levine
    try:
        for i in range(n):
            r = float(math.sin(0.5 * math.pi * (nn - i) / nn) ** 1.5 * Rt)
            if i == 0:
                x_init = 0.0
            else:
                x_prev, r_prev, M_prev, th_prev = wall_first[i - 1]
                mu_prev = math.asin(1.0 / max(M_prev, 1.000001))
                slope = math.tan(th_prev - mu_prev)
                x_init = x_prev + (r - r_prev) / slope if abs(slope) > 1e-12 else x_prev
                x_init = max(x_init, x_prev)
            tk.kliegel_levine = kl_push
            x_final, st_push = _push_throat_point_to_supersonic(r_target=r, x_initial=x_init, Rt=Rt, Rd=Ru,
                                                                gamma=gamma, M_min=M_MIN)
            tk.kliegel_levine = orig
            st_out = kl_out(r / Rt, x_final / Rt, gamma, Ru / Rt) if out_basis != push_basis else st_push
            M = max(st_out.M, 1.0 + 1e-4)
            wall_first.append((x_final, r, M, st_out.theta))
            ms = kl_star(r / Rt, x_final / Rt, gamma, Ru / Rt).M
            info.append({"r": r, "x_init": x_init, "x_final": x_final, "moved": x_final != x_init,
                         "M_push_basis": st_push.M, "M_star": ms, "M_true": mach_from_mstar(ms, gamma),
                         "push_target_met": st_push.M >= M_MIN})
    finally:
        tk.kliegel_levine = orig
    pts = []
    for x, r, M, th in reversed(wall_first):
        nu = prandtl_meyer(M, gamma)
        mu = mach_angle(M)
        pts.append(CharPoint(x=x, r=r, theta=th, M=M, nu=nu, mu=mu, compat_plus=th + nu, compat_minus=th - nu))
    return pts, info


def identity_checks():
    from lrekit.components.nozzle.solvers.nasa import _make_throat_initial_line
    from tests.experiments.nozzle.moc_ext_e09_kl_variants import patched
    out = {}
    for v in ("V0", "V1", "V2", "V3"):
        for n in N_LIST:
            with patched(v):
                ref = _make_throat_initial_line(1.0, RU, 0.5, GAMMA, n, "kliegel_levine")
            loc, _ = build_start_line(n, *VARIANTS[v])
            same = len(ref) == len(loc) and all(
                getattr(p, f) == getattr(q, f) for p, q in zip(ref, loc)
                for f in ("x", "r", "theta", "M", "nu", "mu", "compat_plus", "compat_minus"))
            out[f"{v}_n{n}"] = same
    return out


def geometry(pts):
    wf = list(reversed(pts))
    x = np.array([p.x for p in wf])
    r = np.array([p.r for p in wf])
    dx, dr = np.diff(x), np.diff(r)
    phi = np.degrees(np.arctan2(dr, dx))
    kink = np.abs(np.diff(phi))
    ik = int(np.argmax(kink)) if kink.size else 0
    return {"x_monotone_wall_to_axis": bool(np.all(dx >= 0.0)), "min_dx": float(dx.min()),
            "max_kink_deg": float(kink.max()) if kink.size else 0.0,
            "max_kink_at_r": float(r[ik + 1]) if kink.size else None, "segment_angle_deg": phi}


def spacelike(pts):
    """Signed angular margin (deg) of each start-line segment to the characteristic directions theta -/+ mu.

    Positive: segment outside the open sector (theta-mu, theta+mu) (spacelike); negative: inside it (timelike)."""
    wf = list(reversed(pts))
    res = []
    for a, b in zip(wf[:-1], wf[1:]):
        psi = math.degrees(math.atan2(b.r - a.r, b.x - a.x))
        th = math.degrees(0.5 * (a.theta + b.theta))
        mu = math.degrees(0.5 * (a.mu + b.mu))
        rel = (psi - (th - mu)) % 180.0
        if 0.0 < rel < 2.0 * mu:
            m = -min(rel, 2.0 * mu - rel)
            near = "C-" if rel < mu else "C+"
        else:
            d_minus = min(rel, 180.0 - rel)
            d_plus = abs(rel - 2.0 * mu)
            m = min(d_minus, d_plus)
            near = "C-" if d_minus <= d_plus else "C+"
        res.append((m, near, 0.5 * (a.r + b.r), 0.5 * (a.x + b.x)))
    m = np.array([t[0] for t in res])
    i = int(np.argmin(m))
    return {"min_signed_margin_deg": float(m[i]), "nearest_characteristic": res[i][1], "at_r": res[i][2],
            "at_x": res[i][3], "n_timelike_segments": int(np.sum(m < 0)), "n_segments": len(res),
            "margin_deg": m, "seg_r": np.array([t[2] for t in res])}


def discharge(pts, gamma=GAMMA):
    from lrekit.components.nozzle.solvers.nasa import MOCNode, calc_massflow_along_rrc
    wf = [MOCNode.from_char_point(p, gamma) for p in reversed(pts)]
    mdot = float(calc_massflow_along_rrc(wf, gamma)[0])
    mdot_1d = math.pi * (2.0 / (gamma + 1.0)) ** (1.0 / (gamma - 1.0)) * math.sqrt(gamma * 2.0 / (gamma + 1.0))
    return mdot / mdot_1d


def failing_row_stage(net, theta_B, dth, gamma=GAMMA):
    """Re-run the next row's unit processes (wall, interior, axis closure) and report the first failing stage."""
    from lrekit.components.nozzle.solvers.nasa import (ArcWall, _calc_arc_wall_point_with_special, _calc_axial_mesh_point,
                                 _calc_interior_mesh_point)
    arc = ArcWall(1.0, 0.382, theta_B)
    prev = net.rows[-1]
    res = _calc_arc_wall_point_with_special(prev, arc, gamma, math.radians(dth))
    if res is None:
        return {"stage": "arc wall point", "i": 0}
    wpt, sflag = res
    if wpt.r < 0.0 or wpt.M < 1.0001:
        return {"stage": "wall point state", "i": 0}
    new = [wpt]
    i_last = len(prev) - 1
    for i in range(1, i_last + (1 if sflag else 0)):
        pt, neg = _calc_interior_mesh_point(prev, new, i, sflag, gamma)
        if neg:
            break
        q = prev[min(i if sflag else i + 1, i_last)]
        if pt is None or pt.M < 1.0001:
            return {"stage": "interior unit process " + ("returned None" if pt is None else "subsonic"), "i": i,
                    "ref_point": q}
        if pt.x < new[i - 1].x - 1e-12 * max(abs(new[i - 1].x), 1.0):
            return {"stage": "C- intersection upstream of parent", "i": i, "ref_point": pt}
        new.append(pt)
    if _calc_axial_mesh_point(new, gamma) is None:
        return {"stage": "axis closure (CalcAxialMeshPoint) returned None", "i": len(new), "ref_point": new[-1]}
    return {"stage": "not reproduced"}


def first_rows(pts, n):
    from lrekit.components.nozzle.solvers.bounded import march_bounded_net
    from lrekit.components.nozzle.solvers.nasa import ArcWall
    theta_B = math.radians(THETA_B_DEG)
    arc = ArcWall(1.0, 0.382, theta_B)
    dth = DTHETA_FOR_N[n]
    net = march_bounded_net(pts, arc, GAMMA, arc=arc, dtheta_limit=math.radians(dth), mass_tol=0.02,
                            on_invalid="stop", x_exit=arc.x_end, max_rrcs=N_FIRST_ROWS)
    out = {"stop_reason": net.stop_reason, "rows_stored": net.n_rows, "dtheta_deg": dth}
    if net.stop_reason == "rrc_coalescence":
        fx, fr = net.fold_location
        prev = net.rows[-1]
        d = [math.hypot(p.x - fx, p.r - fr) for p in prev]
        k = int(np.argmin(d))
        nb = [prev[j] for j in (k - 1, k + 1) if 0 <= j < len(prev)]
        sp = float(np.mean([math.hypot(prev[k].x - q.x, prev[k].r - q.r) for q in nb]))
        out.update({"failure": "fold (reversed cell)", "row": net.n_rows, "x": fx, "r": fr,
                    "local_spacing": sp, "r_over_spacing": fr / sp,
                    "point_index_from_wall": k, "row_points": len(prev)})
    elif net.stop_reason == "invalid_cell":
        det = failing_row_stage(net, theta_B, dth)
        prev = net.rows[-1]
        q = det.get("ref_point")
        out.update({"failure": det["stage"], "row": net.n_rows, "point_index_from_wall": det.get("i"),
                    "row_points": len(prev)})
        if q is not None:
            d = [math.hypot(p.x - q.x, p.r - q.r) for p in prev]
            k = int(np.argmin(d))
            nb = [prev[j] for j in (k - 1, k + 1) if 0 <= j < len(prev)]
            sp = float(np.mean([math.hypot(prev[k].x - p.x, prev[k].r - p.r) for p in nb]))
            out.update({"x": float(q.x), "r": float(q.r), "local_spacing": sp, "r_over_spacing": float(q.r) / sp})
    else:
        out["failure"] = None
    return out


def main():
    run = C.Run("E09b", __file__)
    ident = identity_checks()
    print("identity", ident)
    res = {}
    lines = {}
    for v, (sign, pb, ob) in VARIANTS.items():
        for n in N_LIST:
            pts, info = build_start_line(n, sign, pb, ob)
            g = geometry(pts)
            sl = spacelike(pts)
            cd = discharge(pts)
            fr = first_rows(pts, n)
            moved = [d for d in info if d["moved"]]
            res[f"{v}_n{n}"] = {
                "variant": v, "n": n, "sign": sign, "push_basis": pb, "output_basis": ob,
                "geometry": {k: val for k, val in g.items() if k != "segment_angle_deg"},
                "spacelike": {k: val for k, val in sl.items() if k not in ("margin_deg", "seg_r")},
                "push": {"basis_of_1.05_threshold": "M* = hypot(U,V)" if pb == "mstar" else "M from M*",
                         "n_moved": len(moved), "n_points": len(info),
                         "moved_r_range": [min(d["r"] for d in moved), max(d["r"] for d in moved)] if moved else None,
                         "max_push_dx": max((d["x_final"] - d["x_init"] for d in moved), default=0.0),
                         "all_targets_met": all(d["push_target_met"] for d in info),
                         "min_M_star": min(d["M_star"] for d in info), "min_M_true": min(d["M_true"] for d in info)},
                "C_D": cd, "C_D_gt_1": bool(cd > 1.0),
                "first_rows": fr,
            }
            lines[f"{v}_n{n}"] = {"x": np.array([p.x for p in pts]), "r": np.array([p.r for p in pts]),
                                  "M": np.array([p.M for p in pts]), "theta": np.array([p.theta for p in pts]),
                                  "margin": sl["margin_deg"], "seg_r": sl["seg_r"],
                                  "push_r": np.array([d["r"] for d in info]),
                                  "push_dx": np.array([d["x_final"] - d["x_init"] for d in info]),
                                  "push_M": np.array([d["M_push_basis"] for d in info])}
            print(v, n, round(cd, 5), g["x_monotone_wall_to_axis"], round(g["max_kink_deg"], 3),
                  round(sl["min_signed_margin_deg"], 4), sl["nearest_characteristic"], len(moved),
                  fr["stop_reason"], fr.get("row"), fr.get("r_over_spacing"))
    np.savez_compressed(run.out_dir / "e09b_lines.npz", **{f"{k}_{f}": a for k, d in lines.items() for f, a in d.items()})
    figs = make_figure(run, res, lines)
    ok_ident = all(ident.values())
    status = "PASS-DIAGNOSTIC" if ok_ident else "FAIL-DIAGNOSTIC"
    result = {
        "status": status,
        "labels": ["DIAGNOSTIC", "local coefficient and normalisation variants only",
                   "D-K0-BVP not approved (user decision D10)"],
        "evidence_class": "neither (start-line construction review); identity checks are verification",
        "configuration": {
            "case": {"Ru_over_Rt": RU, "gamma": GAMMA, "M_min": M_MIN, "n": list(N_LIST)},
            "variants": DESC,
            "builder": "local copy of nasa_moc._make_throat_initial_line ('kliegel_levine' branch) with separate push "
                       "and output Mach bases; push = production _push_throat_point_to_supersonic with kliegel_levine "
                       "monkeypatched to the push-basis variant; bit-identical to production (patched) for V0-V3",
            "spacelike_margin": "signed angle (deg) between each start-line segment and the nearer of the theta-mu / "
                                "theta+mu directions at the segment midpoint; positive outside the open timelike sector "
                                "(theta-mu, theta+mu), negative inside; mu from the stored M (the M the MOC uses)",
            "kink": "max absolute change of segment direction between consecutive segments (wall to axis)",
            "first_rows": f"production march_bounded_net on the throat arc (Rd 0.382, theta_B {THETA_B_DEG} deg), "
                          f"mass_tol 0.02, on_invalid 'stop', max_rrcs {N_FIRST_ROWS}; dtheta per n {DTHETA_FOR_N} "
                          "(the t18/t19 level pairing); spacing = mean distance to the two neighbours in the last row",
            "C_D": "start-line mass flux (nasa_moc.calc_massflow_along_rrc) / pi rho* a* Rt^2; flag C_D > 1. The bound "
                   "C_D <= 1 is a derivation from anderson Eq. 5.20 / Eq. 5.15 (PDF 220: A >= A* in isentropic flow, "
                   "so rho V <= rho* a*), not a page quote",
        },
        "metrics": {"identity_vs_production": ident, "cases": res},
        "hypotheses": {},
        "figures": figs,
        "caption_facts": {
            "configuration": ("KL start lines for Nozzle-B (Ru/Rt 1.5, gamma 1.23, n = 24/48/64) under the repo series "
                              "(V0), the negated v3 z*y^3 term (V1), M* -> M (V2, V3) and two swapped push/output bases "
                              "(V1a, V1b): shape, spacelike margin, M_min push, near-axis first-row failures and C_D."),
            "evidence_class": "neither (start-line construction review); DIAGNOSTIC",
            "not_established": "The original KL-1969 start line; whether any variant passes the BVP gates; physical C_D.",
        },
        "not_established": "The original KL-1969 start line; whether any variant passes the BVP gates (needs D10).",
        "notes": ["V1a/V1b interpretation: one basis swapped at a time relative to V1 (V1a output basis, V1b push basis).",
                  "No start-line point is moved by the M_min push for Nozzle-B at any n or variant (every point already "
                  "has M >= 1.05 in its push basis at x_init), so V1a reproduces V3 and V1b reproduces V1 exactly; the "
                  "M*/M question changes the output Mach, not the push.",
                  "The start line is built by stepping each point along the previous point's C- direction (NASA "
                  "CalcInitialThroatLine; TT' is RRC 0 of the net), so it is a discretised C- characteristic: the "
                  "spacelike margin is O(1/n) and of either sign, not a positive spacelike clearance."],
    }
    run.finish(result, extra_scripts=[C.REPO / "tests/experiments/nozzle/moc_ext_e09_kl_variants.py",
                                      C.REPO / "tests/experiments/nozzle/moc_ext_e07_forward_replay.py"])
    print("E09b", status)


def make_figure(run, res, lines):
    plt = C.setup_style()
    from matplotlib.gridspec import GridSpec
    col = {"V0": C.COLORS["ink"], "V1": C.COLORS["teal"], "V2": C.COLORS["accent"], "V3": C.COLORS["muted"],
           "V1a": C.COLORS["teal"], "V1b": C.COLORS["teal"]}
    ls = {"V0": "-", "V1": "-", "V2": "--", "V3": ":", "V1a": "--", "V1b": ":"}
    mk = {"V0": "o", "V1": "s", "V2": "^", "V3": "v", "V1a": "D", "V1b": "P"}
    fig = plt.figure(figsize=(C.FULL_W, 7.2))
    gs = GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.34, left=0.1, right=0.98, top=0.97, bottom=0.07)
    ax = fig.add_subplot(gs[0, 0])
    for v in VARIANTS:
        d = lines[f"{v}_n64"]
        ax.plot(d["x"], d["r"], color=col[v], ls=ls[v], lw=1.0, label=v)
    ax.set_xlabel("start-line x / R$_t$ (–)")
    ax.set_ylabel("r / R$_t$ (–)")
    ax.grid(True)
    ax.legend(loc="upper right", fontsize=6.3, ncol=2)
    C.panel_label(ax, "a")
    ax = fig.add_subplot(gs[0, 1])
    for v in VARIANTS:
        d = lines[f"{v}_n64"]
        ax.plot(d["seg_r"], d["margin"], color=col[v], ls=ls[v], lw=1.0, label=v)
    ax.axhline(0, color=C.COLORS["muted"], lw=0.6)
    ax.set_xlabel("r / R$_t$ (segment midpoint, n = 64) (–)")
    ax.set_ylabel("spacelike margin (deg)")
    ax.grid(True)
    C.panel_label(ax, "b")
    ax = fig.add_subplot(gs[1, 0])
    for v in VARIANTS:
        d = lines[f"{v}_n64"]
        ax.plot(d["push_r"], d["push_M"], color=col[v], ls=ls[v], lw=1.0, label=v)
    ax.axhline(M_MIN, color=C.COLORS["accent"], lw=0.8, ls="--")
    ax.text(0.98, M_MIN, "M$_{min}$ = 1.05", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=6.5, color=C.COLORS["accent"])
    ax.set_xlabel("r / R$_t$ on the start line (n = 64) (–)")
    ax.set_ylabel("Mach in the push basis at x$_{init}$ (–)")
    ax.grid(True)
    C.panel_label(ax, "c")
    ax = fig.add_subplot(gs[1, 1])
    for v in VARIANTS:
        ax.plot(N_LIST, [res[f"{v}_n{n}"]["C_D"] for n in N_LIST], color=col[v], ls=ls[v], marker=mk[v], ms=3.5,
                label=v)
    ax.axhline(1.0, color=C.COLORS["accent"], lw=0.8, ls="--")
    ax.set_xticks(N_LIST)
    ax.set_xlabel("start-line points n (–)")
    ax.set_ylabel("start-line C$_D$ = ṁ / ṁ$_{1D}$ (–)")
    ax.grid(True)
    C.panel_label(ax, "d")
    ax = fig.add_subplot(gs[2, :])
    names = list(VARIANTS)
    for jv, v in enumerate(names):
        for jn, n in enumerate(N_LIST):
            fr = res[f"{v}_n{n}"]["first_rows"]
            xpos = jv + (jn - 1) * 0.22
            if fr.get("failure") is None:
                ax.plot(xpos, N_FIRST_ROWS + 2, marker="o", ms=4, mfc="none", mec=C.COLORS["muted"], ls="none")
            else:
                mark = "x" if "fold" in str(fr.get("failure")) else "s"
                ax.plot(xpos, fr.get("row", 0), marker=mark, ms=5, color=col[v], ls="none")
                if fr.get("r_over_spacing") is not None:
                    ax.annotate(f"{fr['r_over_spacing']:.1f}", (xpos, fr.get("row", 0)), textcoords="offset points",
                                xytext=(0, 5), ha="center", fontsize=5.8, color=C.COLORS["ink"])
    ax.axhline(N_FIRST_ROWS + 1, color=C.COLORS["muted"], lw=0.5, ls=":")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{v}\n(n 24 / 48 / 64)" for v in names], fontsize=6.5)
    ax.set_ylabel("first failing RRC index (–)")
    ax.plot([], [], marker="x", color=C.COLORS["ink"], ls="none", label="fold (label: r / local spacing)")
    ax.plot([], [], marker="s", color=C.COLORS["ink"], ls="none", label="unit process failed")
    ax.plot([], [], marker="o", mfc="none", mec=C.COLORS["muted"], ls="none", label=f"no failure in {N_FIRST_ROWS} RRCs")
    ax.legend(loc="upper left", fontsize=6.3, ncol=3)
    ax.set_ylim(0, N_FIRST_ROWS + 8)
    ax.grid(True, axis="y")
    C.panel_label(ax, "e")
    return [C.save_figure(fig, run.out_dir, "e09b_startline_review")]


if __name__ == "__main__":
    main()
