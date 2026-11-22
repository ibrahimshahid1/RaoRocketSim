"""E08: FSS separation-onset criteria on the resolved Nozzle-B wall, swept over p0/pa (screening comparison).

Opt-in experiment script (moc_extensions_plan_2026-11-21). Report-only; no physics pass/fail.
Wall states: BVP BDE wall of the banked t19 solutions (the E07 forward replay folded, see E07 result).
Run: .venv-jax/bin/python -m tests.experiments.nozzle.moc_ext_e08_separation_sweep
"""

from __future__ import annotations

import math

import numpy as np

from scripts import moc_ext_e00_common as C

P0_OVER_PA = np.geomspace(5.0, 300.0, 60)
SPOT = ((10.0, 2.0), (50.0, 2.6), (200.0, 3.5))
SPOT_TOL = 1.0e-12
RESERVE = 1.2
SPREAD_THRESHOLD = 1.2           # pre-declared: disagreement "more than the 20 % reserve" = max/min > 1.2
SCHMUCKER_DIFF_THRESHOLD = 0.01  # pre-declared: wall-state vs Q1D incipient p0/pa differ by more than 1 %
SCHMUCKER_M_MAX = 5.0
TP1207_M = (2.4, 4.5)

CRITERIA = ("summerfield", "schilling_0582", "schilling_0583", "kalt_badal", "schmucker", "tp1207")
LABEL = {"summerfield": "Summerfield 0.4", "schilling_0582": "Schilling 0.582 (repo)",
         "schilling_0583": "Schilling 0.583 (Schmucker Eq. 9)", "kalt_badal": "Kalt–Badal",
         "schmucker": "Schmucker (local M)", "tp1207": "TP-1207 contoured fit"}


def psep_over_pa(name, p0_over_pa, M):
    """Separation pressure / ambient. M is the local wall Mach (used by the Mach-form criteria)."""
    M = np.asarray(M, dtype=float)
    if name == "summerfield":
        return np.full_like(M, 0.4)
    if name == "schilling_0582":
        return np.full_like(M, 0.582 * p0_over_pa ** -0.195)
    if name == "schilling_0583":
        return np.full_like(M, 0.583 * p0_over_pa ** -0.195)
    if name == "kalt_badal":
        return np.full_like(M, (2.0 / 3.0) * p0_over_pa ** -0.2)
    if name == "schmucker":
        return np.where(1.88 * M - 1.0 > 0.0, np.abs(1.88 * M - 1.0) ** -0.64, np.inf)
    if name == "tp1207":
        return 1.082 - 0.363 * M + 0.0386 * M * M
    raise ValueError(name)


def valid_mask(name, M):
    M = np.asarray(M, dtype=float)
    if name == "schmucker":
        return M <= SCHMUCKER_M_MAX
    if name == "tp1207":
        return (M >= TP1207_M[0]) & (M <= TP1207_M[1])
    return np.ones_like(M, dtype=bool)


def spot_checks():
    from lrekit import separation as S
    rows = []
    for ratio, M in SPOT:
        pa = 1.0e5
        pc = ratio * pa
        pairs = {
            "summerfield": (psep_over_pa("summerfield", ratio, [M])[0], S.summerfield_separation_pressure(pa) / pa),
            "schilling_0582": (psep_over_pa("schilling_0582", ratio, [M])[0], S.schilling_separation_ratio(pc / pa)),
            "kalt_badal": (psep_over_pa("kalt_badal", ratio, [M])[0], S.kalt_badal_separation_ratio(pc / pa)),
            "schmucker": (psep_over_pa("schmucker", ratio, [M])[0],
                          S.schmucker_separation_ratio(M, pa / pc) / (pa / pc)),
        }
        for k, (mine, repo) in pairs.items():
            rel = abs(mine - repo) / abs(repo)
            rows.append({"criterion": k, "p0_over_pa": ratio, "M": M, "script": mine, "repo": repo, "rel": rel,
                         "passes": rel <= SPOT_TOL})
    return rows


def wall_states(sol):
    from lrekit.core.gas_dynamics import isentropic_pressure_ratio
    bde = sol.construction_diagnostics["bde_artifacts"]
    arc = [(r[0].x, r[0].M) for r in bde["kernel"].rrcs]
    bell = [(p.x, p.M) for p in bde["bde_region"].wall_contour]
    xm = np.array(arc + bell, dtype=float)
    order = np.argsort(xm[:, 0], kind="stable")
    xm = xm[order]
    keep = np.concatenate([[True], np.diff(xm[:, 0]) > 0.0])
    xm = xm[keep]
    p = np.array([isentropic_pressure_ratio(m, C.GAMMA) for m in xm[:, 1]])
    return xm[:, 0], xm[:, 1], p


def first_crossing(x, f):
    """x of the first f <= 0 going downstream (linear interpolation); None if never."""
    idx = np.where(f <= 0.0)[0]
    if idx.size == 0:
        return None, None
    i = int(idx[0])
    if i == 0:
        return float(x[0]), 0
    t = f[i - 1] / (f[i - 1] - f[i])
    return float(x[i - 1] + t * (x[i] - x[i - 1])), i


def sweep(x, M, p, name, apply_mask=True):
    xs, flags = [], []
    valid = valid_mask(name, M)
    for ratio in P0_OVER_PA:
        ps = psep_over_pa(name, ratio, M) / ratio
        f = p - ps
        xc, i = first_crossing(x, f)
        if xc is None:
            xs.append(np.nan)
            flags.append("attached")
            continue
        ok = valid[i] and (i == 0 or valid[i - 1])
        if apply_mask and not ok:
            xs.append(np.nan)
            flags.append("outside_validity")
        else:
            xs.append(xc)
            flags.append("separated" if ok else "separated_extrapolated")
    return np.array(xs), flags


def incipient(name, M_e, p_e):
    """p0/pa at which p_e = p_sep at the exit station (separation onset at the lip)."""
    if not valid_mask(name, [M_e])[0]:
        return float("nan")
    if name in ("summerfield", "schmucker", "tp1207"):
        return float(psep_over_pa(name, 1.0, [M_e])[0] / p_e)
    k1, k2 = {"schilling_0582": (0.582, -0.195), "schilling_0583": (0.583, -0.195),
              "kalt_badal": (2.0 / 3.0, -0.2)}[name]
    return float((k1 / p_e) ** (1.0 / (1.0 - k2)))


def design_ratio(name, M_e, p_e, reserve=RESERVE):
    from scipy.optimize import brentq
    if not valid_mask(name, [M_e])[0]:
        return float("nan")
    g = lambda r: p_e - reserve * psep_over_pa(name, r, [M_e])[0] / r
    return float(brentq(g, 1.0001, 1.0e4, xtol=1e-12))


def main():
    from lrekit.core.gas_dynamics import isentropic_pressure_ratio, mach_from_area_ratio
    import lrekit.numerics.jax.thermal as jt

    run = C.Run("E08", __file__)
    report = C.load_report()
    spots = spot_checks()
    spot_ok = all(s["passes"] for s in spots)

    eps = float(report["refined"]["config"]["epsilon"])
    Me_q = mach_from_area_ratio(eps, C.GAMMA, supersonic=True)
    pe_q = isentropic_pressure_ratio(Me_q, C.GAMMA)

    lv = {}
    for lvl in ("refined", "baseline"):
        run.add_input(C.PKL[lvl], report["artifact_sha256"][f"{lvl}_solution.pkl"])
        sol = C.load_solution(lvl)
        x, M, p = wall_states(sol)
        crit = {}
        for name in CRITERIA:
            xs, flags = sweep(x, M, p, name)
            xs_ext, _ = sweep(x, M, p, name, apply_mask=False)
            crit[name] = {
                "x_sep": xs, "x_sep_unmasked": xs_ext, "flags": flags,
                "n_outside_validity": int(sum(f == "outside_validity" for f in flags)),
                "n_wall_stations_outside_validity": int(np.sum(~valid_mask(name, M))),
                "incipient_wall": incipient(name, M[-1], p[-1]),
                "incipient_q1d": incipient(name, Me_q, pe_q),
                "design_wall": design_ratio(name, M[-1], p[-1]),
                "design_q1d": design_ratio(name, Me_q, pe_q),
            }
        lv[lvl] = {"x": x, "M": M, "p": p, "criteria": crit, "n_stations": int(len(x)),
                   "monotone_p": bool(np.all(np.diff(p) <= 0.0)), "M_exit": float(M[-1]), "p_exit": float(p[-1])}
        del sol

    ref = lv["refined"]
    inc_wall = {n: ref["criteria"][n]["incipient_wall"] for n in CRITERIA}
    main_set = ("summerfield", "schilling_0582", "kalt_badal", "schmucker", "tp1207")
    vals = np.array([inc_wall[n] for n in main_set])
    spread = float(np.nanmax(vals) / np.nanmin(vals))
    sch_diff = (ref["criteria"]["schmucker"]["incipient_wall"] - ref["criteria"]["schmucker"]["incipient_q1d"]) \
        / ref["criteria"]["schmucker"]["incipient_q1d"]

    def grid_consistency(name):
        xs = ref["criteria"][name]["x_sep"]
        inc = inc_wall[name]
        if not math.isfinite(inc):
            return None
        above = P0_OVER_PA > inc * (1 + 1e-9)
        below = P0_OVER_PA < inc * (1 - 1e-9)
        return {"attached_above_incipient": bool(np.all(np.isnan(xs[above]))),
                "separated_below_incipient": bool(np.all(np.isfinite(ref["criteria"][name]["x_sep_unmasked"][below])))}
    consistency = {n: grid_consistency(n) for n in CRITERIA}

    margins = {"p0_over_pa": P0_OVER_PA,
               "mdo_q1d": np.array([float(jt.schmucker_separation_margin(eps, C.GAMMA, 1.0, 1.0 / r)) - RESERVE
                                    for r in P0_OVER_PA]),
               "wall_exit": np.array([ref["p_exit"] / (psep_over_pa("schmucker", r, [ref["M_exit"]])[0] / r) - RESERVE
                                      for r in P0_OVER_PA])}

    figs = make_figure(run, lv, margins) if spot_ok else []
    status = "PASS" if spot_ok else "FAIL"
    metrics = {
        "spot_checks": spots,
        "q1d_exit": {"Me": Me_q, "pe_over_p0": pe_q},
        "wall_exit": {lvl: {"M": v["M_exit"], "p_over_p0": v["p_exit"], "n_stations": v["n_stations"],
                            "p_monotone_decreasing": v["monotone_p"]} for lvl, v in lv.items()},
        "incipient_p0_over_pa": {lvl: {n: {"wall": v["criteria"][n]["incipient_wall"],
                                           "q1d": v["criteria"][n]["incipient_q1d"],
                                           "design_wall_reserve_1.2": v["criteria"][n]["design_wall"],
                                           "design_q1d_reserve_1.2": v["criteria"][n]["design_q1d"]}
                                       for n in CRITERIA} for lvl, v in lv.items()},
        "validity_counts": {lvl: {n: {"grid_points_outside_validity": v["criteria"][n]["n_outside_validity"],
                                      "wall_stations_outside_validity": v["criteria"][n]["n_wall_stations_outside_validity"]}
                                  for n in CRITERIA} for lvl, v in lv.items()},
        "spread_incipient_wall_max_over_min": spread,
        "spread_members": list(main_set),
        "schmucker_wall_vs_q1d_incipient_rel": sch_diff,
        "schilling_0583_over_0582": 0.583 / 0.582,
        "level_cross_check_incipient_rel": {n: (lv["baseline"]["criteria"][n]["incipient_wall"] - inc_wall[n]) / inc_wall[n]
                                            for n in CRITERIA if math.isfinite(inc_wall[n])},
        "grid_consistency": consistency,
        "x_sep_grid": {"p0_over_pa": P0_OVER_PA, **{n: ref["criteria"][n]["x_sep"] for n in CRITERIA}},
        "schmucker_margin_curves": margins,
    }
    result = {
        "status": status,
        "evidence_class": "neither (screening-criteria comparison on an inviscid wall)",
        "configuration": {
            "wall_states": "BVP BDE wall of the 48/64 t19 solution (kernel arc wall points + bde_region.wall_contour), "
                           "p/p0 isentropic at gamma 1.23; 32/48 used as the cross-check",
            "primary_input_deviation": "spec primary input (E07 forward wall states) unavailable: E07 forward net folded "
                                       "downstream of the exit plane on the first post-kernel RRC; see b1_outbox request",
            "p0_over_pa_grid": "geomspace(5, 300, 60)",
            "criteria": {n: LABEL[n] for n in CRITERIA},
            "sources_checked": {
                "summerfield/schilling/kalt_badal/schmucker": "Schmucker NASA TM-77396 (1984 translation) Eqs. 7-11, "
                                                              "PDF text read; repo forms via Ostlund Eqs. 28-30",
                "tp1207": "Morrisette & Goldberg NASA TP-1207 Eq. (10), fit range 2.4 <= Ms <= 4.5, PDF text read",
            },
            "masks": {"schmucker": f"M_i <= {SCHMUCKER_M_MAX}", "tp1207": f"{TP1207_M[0]} <= M_s <= {TP1207_M[1]}"},
            "separation_location": "first wall station going downstream with p_w <= p_sep (Mach-form criteria use the "
                                   "local wall M); linear interpolation between stations",
            "incipient": "p0/pa where p_w = p_sep at the exit wall node (closed form; Schilling/Kalt-Badal solved "
                         "for their p0/pa dependence)",
            "design_reserve": "smallest p0/pa with p_e >= 1.2 p_sep at the exit station",
            "mdo_margin": "lrekit.numerics.jax.thermal.schmucker_separation_margin(eps, gamma, 1, pa/p0) - 1.2 (Q1D exit)",
            "predeclared": {"SPREAD_THRESHOLD": SPREAD_THRESHOLD, "SCHMUCKER_DIFF_THRESHOLD": SCHMUCKER_DIFF_THRESHOLD,
                            "SPOT_TOL": SPOT_TOL},
        },
        "metrics": metrics,
        "hypotheses": {
            "H_criteria_spread_exceeds_reserve": "supported" if spread > SPREAD_THRESHOLD else "not supported",
            "H_schmucker_wall_differs_from_mdo_q1d": "supported" if abs(sch_diff) > SCHMUCKER_DIFF_THRESHOLD
            else "not supported",
        },
        "figures": figs,
        "caption_facts": {
            "configuration": ("Five FSS onset criteria (Summerfield, Schilling, Kalt–Badal, Schmucker, TP-1207 fit) "
                              "applied to the inviscid BVP wall states of the accepted t19 Nozzle-B (48/64; gamma 1.23, "
                              "epsilon 4.973, a vacuum design used only as a wall-state carrier) over p0/pa = 5-300."),
            "evidence_class": "neither (screening-criteria comparison on an inviscid wall)",
            "not_established": "Which criterion is physically right; RSS, hysteresis or side loads; any validation.",
        },
        "not_established": "Which criterion is physically right; RSS, hysteresis or side loads; any validation.",
        "notes": [
            "Nozzle-B is a vacuum design; it is used here only as a resolved wall-state carrier.",
            "Schmucker (TM-77396 p. 37) states Eq. 9 (Schilling) gives far too small separation pressures and should "
            "not be used for chemical engines; Summerfield 0.4 is stated as a guideline for pc/pa about 15-20.",
            "Schilling 0.582 (repo, via Ostlund Eq. 29) vs 0.583 (Schmucker Eq. 8/9): 0.17 % coefficient difference, "
            "flagged only, no change.",
        ],
    }
    run.finish(result)
    print("E08", status, "spread", spread, "sch_diff", sch_diff)
    for n in CRITERIA:
        print(f"  {n:16s} inc_wall {inc_wall[n]:10.4f} inc_q1d {ref['criteria'][n]['incipient_q1d']:10.4f} "
              f"design_wall {ref['criteria'][n]['design_wall']:10.4f}")
    return 0 if spot_ok else 1


def make_figure(run, lv, margins):
    plt = C.setup_style()
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
    ref = lv["refined"]
    style = {"summerfield": (C.COLORS["ink"], None), "schilling_0582": (C.COLORS["teal"], None),
             "schilling_0583": (C.COLORS["teal"], "|"), "kalt_badal": (C.COLORS["muted"], None),
             "schmucker": (C.COLORS["accent"], None), "tp1207": (C.COLORS["ink"], "o")}
    fig = plt.figure(figsize=(C.FULL_W, 5.6))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.0], hspace=0.45, wspace=0.36,
                  left=0.1, right=0.98, top=0.96, bottom=0.08)

    ax = fig.add_subplot(gs[0, 0])
    for n in CRITERIA:
        c, mk = style[n]
        xs = ref["criteria"][n]["x_sep"]
        xe = ref["criteria"][n]["x_sep_unmasked"]
        ax.plot(P0_OVER_PA, xs, ls="-", color=c, lw=1.1, marker=mk, ms=2.8, markevery=2, label=LABEL[n])
        inval = np.isnan(xs) & np.isfinite(xe)
        if inval.any():
            ax.plot(P0_OVER_PA, np.where(inval | np.roll(inval, 1) | np.roll(inval, -1), xe, np.nan), ls="--",
                    color=c, lw=0.9)
        inc = ref["criteria"][n]["incipient_wall"]
        if math.isfinite(inc):
            ax.plot([inc], [ref["x"][-1]], marker="v", ms=3.5, color=c, ls="none")
    ax.set_xlim(4.8, 12.0)
    ax.set_ylim(0.8, 3.05)
    ax.axhline(2.94, color=C.COLORS["muted"], lw=0.6, ls=":")
    ax.text(11.9, 2.96, "exit", ha="right", va="bottom", fontsize=6.5, color=C.COLORS["muted"])
    ax.set_xlabel("p$_0$ / p$_a$ (–)")
    ax.set_ylabel("separation station x$_{sep}$ / R$_t$ (–)")
    ax.grid(True)
    C.panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(margins["p0_over_pa"], margins["mdo_q1d"], color=C.COLORS["ink"], lw=1.1, label="MDO: Q1D exit state")
    ax.plot(margins["p0_over_pa"], margins["wall_exit"], color=C.COLORS["accent"], lw=1.1, label="MOC wall exit state")
    ax.axhline(0.0, color=C.COLORS["muted"], lw=0.7, ls="--")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([5, 10, 20, 50, 100, 300]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("p$_0$ / p$_a$ (–)")
    ax.set_ylabel("Schmucker p$_e$/p$_{sep}$ − 1.2 (–)")
    ax.grid(True, which="major")
    ax.legend(loc="upper left", fontsize=6.5)
    C.panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, :])
    names = list(CRITERIA)
    y = np.arange(len(names))[::-1]
    for yi, n in zip(y, names):
        c = style[n][0]
        iw = ref["criteria"][n]["incipient_wall"]
        dw = ref["criteria"][n]["design_wall"]
        iq = ref["criteria"][n]["incipient_q1d"]
        if math.isfinite(iw):
            ax.barh(yi + 0.18, iw, height=0.32, color=c, alpha=0.9)
            ax.barh(yi + 0.18, dw - iw, left=iw, height=0.32, color=c, alpha=0.3)
        if math.isfinite(iq):
            ax.barh(yi - 0.18, iq, height=0.32, color="none", edgecolor=c, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[n] for n in names], fontsize=6.8)
    ax.set_xlabel("incipient-separation p$_0$ / p$_a$ (–)")
    ax.legend(handles=[Patch(color=C.COLORS["muted"], alpha=0.9, label="MOC wall exit state"),
                       Patch(color=C.COLORS["muted"], alpha=0.3, label="extension to the 1.2 reserve"),
                       Patch(facecolor="none", edgecolor=C.COLORS["muted"], label="Q1D exit state")],
              loc="center right", fontsize=6.5)
    ax.set_xlim(0, 20)
    ax.grid(True, axis="x")
    C.panel_label(ax, "b")
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.axes[0].legend(handles, labels, loc="lower right", fontsize=5.8, handlelength=1.6)
    pos = fig.axes[2].get_position()
    fig.axes[2].set_position([0.33, pos.y0, 0.98 - 0.33, pos.height])
    return [C.save_figure(fig, run.out_dir, "e08_separation_sweep")]


if __name__ == "__main__":
    raise SystemExit(main())
