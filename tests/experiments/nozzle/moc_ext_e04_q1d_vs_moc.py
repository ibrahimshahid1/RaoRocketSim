"""E04: quasi-1-D closure vs MOC thrust coefficient and exit pressure for the accepted Nozzle-B.

Opt-in experiment script (moc_extensions_plan_2026-11-21). Architecture scoping; report-only.
Run: .venv-jax/bin/python -m tests.experiments.nozzle.moc_ext_e04_q1d_vs_moc
"""

from __future__ import annotations

import math

import numpy as np

from scripts import moc_ext_e00_common as C

REL_TOL = 1.0e-10
P0_OVER_PA = (20.0, 50.0, 100.0)
SP8120_RESERVE = 1.2
PRESSURE_DIFF_THRESHOLD = 0.01  # pre-declared: "differs" means |p_wall - Pe| / Pe > 1 %


def cde_averages(nodes, gamma):
    from lrekit.core.gas_dynamics import (isentropic_density_ratio, isentropic_pressure_ratio,
                                     isentropic_temperature_ratio)
    wm = wa = pm = pa = 0.0
    for p0, p1 in zip(nodes[:-1], nodes[1:]):
        dx, dr = float(p1.x - p0.x), float(p1.r - p0.r)
        ds = math.hypot(dx, dr)
        if ds <= 1e-12:
            continue
        beta = math.atan2(dr, dx)
        M = max(0.5 * (float(p0.M) + float(p1.M)), 1.000001)
        th = 0.5 * (float(p0.theta) + float(p1.theta))
        r = max(0.5 * (float(p0.r) + float(p1.r)), 1e-12)
        p = isentropic_pressure_ratio(M, gamma)
        V = M * math.sqrt(gamma * isentropic_temperature_ratio(M, gamma))
        dm = 2.0 * math.pi * r * isentropic_density_ratio(M, gamma) * V * abs(math.sin(beta - th)) * ds
        da = 2.0 * math.pi * r * abs(dr)
        wm += dm
        pm += p * dm
        wa += da
        pa += p * da
    return {"mass_averaged_p_over_p0": pm / wm, "area_averaged_p_over_p0": pa / wa,
            "mass_flux": wm, "projected_area_over_pi_rt2": wa / math.pi}


def main():
    import lrekit.numerics.jax.thermal as jt
    from lrekit.core.gas_dynamics import isentropic_pressure_ratio, mach_from_area_ratio, thrust_coefficient
    from lrekit.optimization.propellants import PROPELLANTS
    from lrekit.components.nozzle.solvers.nasa import full_control_surface_thrust
    from lrekit.components.nozzle.solvers.variational import _control_surface_flow_nodes
    from lrekit.components.nozzle.separation import schmucker_separation_ratio

    run = C.Run("E04", __file__)
    report = C.load_report()
    run.add_input(C.REPORT)
    cfg = report["refined"]["config"]
    gamma, eps = float(cfg["gamma"]), float(cfg["epsilon"])

    sols = {}
    levels = {}
    for lvl in ("baseline", "refined"):
        run.add_input(C.PKL[lvl], report["artifact_sha256"][f"{lvl}_solution.pkl"])
        s = C.load_solution(lvl)
        sols[lvl] = s
        kernel = s.construction_diagnostics["bde_artifacts"]["kernel"]
        fct = full_control_surface_thrust(kernel, _control_surface_flow_nodes(s.control_surface),
                                          gamma=gamma, Rt=1.0, pa_over_p0=0.0)
        wall_E = s.construction_diagnostics["bde_artifacts"]["bde_region"].wall_contour[-1]
        levels[lvl] = {
            "cf_report": float(report[lvl]["Cf"]),
            "cf_pickle": float(s.thrust_coefficient),
            "cf_recomputed_full_cde": float(fct.cf_cde),
            "cf_cd": float(fct.cf_cd), "cf_de": float(fct.cf_de),
            "cde_mass_residual_rel": float(fct.mass_residual_rel),
            "wall_exit": {"x": float(wall_E.x), "r": float(wall_E.r), "M": float(wall_E.M),
                          "theta_deg": math.degrees(float(wall_E.theta)),
                          "p_over_p0": isentropic_pressure_ratio(float(wall_E.M), gamma)},
            "cde": cde_averages(list(fct.CDE), gamma),
            "cf_ideal_thrust_sanity": float(s.construction_diagnostics["thrust_sanity"]["cf_ideal"]),
        }

    Me = mach_from_area_ratio(eps, gamma, supersonic=True)
    pe = isentropic_pressure_ratio(Me, gamma)
    cf1d_a = thrust_coefficient(Me, gamma, pe, 0.0, eps)
    cf1d_b = float(jt.ambient_thrust_coefficient(eps, gamma, 1.0, 0.0))
    ref = levels["refined"]
    checks = {
        "cf_moc_report_rounds_to_1.57955115": round(ref["cf_report"], 8) == 1.57955115,
        "cf_moc_pickle_vs_report_rel": abs(ref["cf_pickle"] - ref["cf_report"]) / ref["cf_report"],
        "cf_moc_recomputed_vs_report_rel": abs(ref["cf_recomputed_full_cde"] - ref["cf_report"]) / ref["cf_report"],
        "cf_1d_gas_dynamics_vs_jax_rel": abs(cf1d_a - cf1d_b) / cf1d_a,
        "cf_1d_vs_thrust_sanity_cf_ideal_rel": abs(cf1d_a - ref["cf_ideal_thrust_sanity"]) / cf1d_a,
    }
    passes = (checks["cf_moc_report_rounds_to_1.57955115"]
              and checks["cf_moc_pickle_vs_report_rel"] <= REL_TOL
              and checks["cf_moc_recomputed_vs_report_rel"] <= REL_TOL
              and checks["cf_1d_gas_dynamics_vs_jax_rel"] <= REL_TOL)

    lam = {lvl: v["cf_report"] / cf1d_a for lvl, v in levels.items()}
    eta = {k: float(p.eta_CF) for k, p in PROPELLANTS.items()}

    margins = []
    for ratio in P0_OVER_PA:
        pa_pc = 1.0 / ratio
        q_sep = schmucker_separation_ratio(Me, pa_pc)
        mdo_raw = float(jt.schmucker_separation_margin(eps, gamma, 1.0, pa_pc))
        w = ref["wall_exit"]
        w_sep = schmucker_separation_ratio(w["M"], pa_pc)
        margins.append({
            "p0_over_pa": ratio,
            "q1d": {"Me": Me, "pe_over_p0": pe, "psep_over_p0": q_sep, "margin": pe / q_sep - SP8120_RESERVE},
            "mdo_jax_margin": mdo_raw - SP8120_RESERVE,
            "wall": {"M": w["M"], "p_over_p0": w["p_over_p0"], "psep_over_p0": w_sep,
                     "margin": w["p_over_p0"] / w_sep - SP8120_RESERVE},
        })
    mdo_parity = max(abs(m["mdo_jax_margin"] - m["q1d"]["margin"]) for m in margins)

    p_rel = (ref["wall_exit"]["p_over_p0"] - pe) / pe
    hyp = {
        "H_lambda_below_eta_CF_defaults": ("supported" if max(lam.values()) < min(eta.values()) else "not supported"),
        "H_wall_exit_pressure_differs_from_Q1D": ("supported" if abs(p_rel) > PRESSURE_DIFF_THRESHOLD
                                                  else "not supported"),
    }
    status = "PASS" if passes else "FAIL"
    figs = make_figure(run, cf1d_a, levels, eta, pe, margins) if passes else []
    caption_cfg = ("Accepted t19 Nozzle-B (gamma 1.23, epsilon 4.973, 64% length, pa = 0, KL start line, Ru/Rt 1.5); "
                   "MOC values from the banked 32/48 and 48/64 BVP solutions, Q1D from the constant-gamma "
                   "area-Mach relation at the same gamma and epsilon.")
    result = {
        "status": status,
        "evidence_class": "neither (architecture scoping for one constant-gamma case)",
        "configuration": {
            "gamma": gamma, "epsilon": eps, "pa_over_p0": 0.0, "length_pct": cfg["length_pct"],
            "cf_moc_definition": "full C-D-E control-surface integral (nasa_moc.full_control_surface_thrust), normalised by p0*pi*Rt^2",
            "cf_1d_functions": ["lrekit.core.gas_dynamics.thrust_coefficient", "lrekit.numerics.jax.thermal.ambient_thrust_coefficient"],
            "wall_exit_state": "last BDE wall node (bde_region.wall_contour[-1]) of each level",
            "cde_averages": "segment-midpoint states on the C-D-E polyline; mass weight 2*pi*r*rho*V*|sin(beta-theta)|*ds "
                            "(as nasa_moc.curve_mass_flux); area weight 2*pi*r*|dr| (axially projected annulus)",
            "schmucker": "p_sep/pa = (1.88 M - 1)^-0.64 (lrekit.components.nozzle.separation.schmucker_separation_ratio); margin = p/p_sep - 1.2",
            "p0_over_pa": list(P0_OVER_PA),
            "predeclared": {"REL_TOL": REL_TOL, "PRESSURE_DIFF_THRESHOLD": PRESSURE_DIFF_THRESHOLD},
        },
        "metrics": {
            "acceptance_checks": checks,
            "cf_1d": cf1d_a, "cf_1d_jax": cf1d_b, "Me_q1d": Me, "pe_over_p0_q1d": pe,
            "levels": levels,
            "lambda_cf_moc_over_cf_1d": lam,
            "eta_CF_defaults": eta,
            "eta_CF_times_cf_1d": {k: v * cf1d_a for k, v in eta.items()},
            "wall_exit_p_rel_to_q1d": p_rel,
            "cde_mass_avg_p_rel_to_q1d": (ref["cde"]["mass_averaged_p_over_p0"] - pe) / pe,
            "cde_area_avg_p_rel_to_q1d": (ref["cde"]["area_averaged_p_over_p0"] - pe) / pe,
            "schmucker_margins": margins,
            "mdo_jax_vs_numpy_margin_max_abs": mdo_parity,
        },
        "hypotheses": hyp,
        "figures": figs,
        "caption_facts": {
            "configuration": caption_cfg,
            "evidence_class": "neither (architecture scoping for one constant-gamma case)",
            "not_established": "A general correction law; the 80%-length TOP default (forward net folds, E10); "
                               "any viscous, chemical or physical-validation claim.",
        },
        "not_established": "A general correction law; the 80%-length TOP default is not covered because its forward net folds (E10).",
        "notes": [
            "CF_MOC is normalised by p0*pi*Rt^2, so lambda also carries the KL start-line mass-flow (discharge) "
            "difference from ideal 1-D, not only divergence and non-uniformity.",
            "The eta_CF values belong to propellants with their own gamma; here they are only compared as multipliers "
            "on CF_1D at gamma 1.23.",
            "The C-D-E surface is a characteristic, not the exit plane; its averages are surrogates for the exit plane. "
            "The C-D portion starts on the axis upstream (C at x about 0.6 Rt), so the mass average (about 3x Q1D Pe) is "
            "dominated by lower-Mach flow and is not an exit-plane pressure; an exit-plane cross-check from the E07 "
            "forward net is unavailable because that net folded.",
        ],
    }
    run.finish(result)
    print("E04", status, checks)
    print(" lambda", lam, "p_rel", p_rel, "hyp", hyp)
    return 0 if passes else 1


def make_figure(run, cf1d, levels, eta, pe, margins):
    plt = C.setup_style()
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(C.FULL_W, 4.6))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.1, 1.0], hspace=0.5, wspace=0.75,
                  left=0.285, right=0.98, top=0.95, bottom=0.11)
    axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

    ax = axes[0]
    names = {"lox/rp-1": "LOX/RP-1", "lox/lch4": "LOX/LCH4", "lox/lh2": "LOX/LH2", "n2o4/mmh": "N2O4/MMH",
             "n2o/ethanol": "N2O/ethanol"}
    rows = [("C$_F$,1D (ideal)", cf1d, C.COLORS["ink"], "s")]
    rows += [(f"η$_{{CF}}$ {v:.3f} × C$_F$,1D ({names.get(k, k)})", v * cf1d, C.COLORS["muted"], "o")
             for k, v in eta.items()]
    rows += [("C$_F$,MOC 32/48", levels["baseline"]["cf_report"], C.COLORS["teal"], "D"),
             ("C$_F$,MOC 48/64", levels["refined"]["cf_report"], C.COLORS["accent"], "D")]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, val, col, mk) in zip(y, rows):
        ax.plot([val], [yi], marker=mk, color=col, ms=4.5, ls="none")
        ax.hlines(yi, 1.56, val, color="#D5DBDE", lw=0.6, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.set_xlim(1.56, 1.645)
    ax.set_xlabel("vacuum thrust coefficient C$_F$ (–)")
    ax.grid(True, axis="x")
    C.panel_label(ax, "a")

    ax = axes[1]
    ref = levels["refined"]
    base = levels["baseline"]
    prow = [("Q1D exit", pe, C.COLORS["ink"], "s", None),
            ("MOC wall exit node", ref["wall_exit"]["p_over_p0"], C.COLORS["accent"], "D", base["wall_exit"]["p_over_p0"]),
            ("C-D-E, mass avg.", ref["cde"]["mass_averaged_p_over_p0"], C.COLORS["teal"], "o",
             base["cde"]["mass_averaged_p_over_p0"]),
            ("C-D-E, area avg.", ref["cde"]["area_averaged_p_over_p0"], C.COLORS["teal"], "^",
             base["cde"]["area_averaged_p_over_p0"])]
    y = np.arange(len(prow))[::-1]
    for yi, (lab, val, col, mk, bval) in zip(y, prow):
        ax.plot([val], [yi], marker=mk, color=col, ms=4.5, ls="none", label=None)
        if bval is not None:
            ax.plot([bval], [yi], marker=mk, mfc="none", mec=col, ms=6.5, ls="none")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in prow], fontsize=7)
    ax.set_xlim(0.02, 0.1)
    ax.set_ylim(-0.7, len(prow) - 0.4)
    ax.set_xlabel("p / p$_0$ (–)")
    ax.grid(True, axis="x")
    ax.plot([], [], marker="D", color=C.COLORS["muted"], ls="none", ms=4.5, label="48/64")
    ax.plot([], [], marker="D", mfc="none", mec=C.COLORS["muted"], ls="none", ms=6.5, label="32/48")
    ax.legend(loc="upper right", fontsize=6.5, handletextpad=0.2, ncol=2, columnspacing=0.6)
    C.panel_label(ax, "b")

    ax = axes[2]
    r = [m["p0_over_pa"] for m in margins]
    ax.plot(r, [m["q1d"]["margin"] for m in margins], color=C.COLORS["ink"], marker="s", ms=4, label="Q1D exit (MDO)")
    ax.plot(r, [m["wall"]["margin"] for m in margins], color=C.COLORS["accent"], marker="D", ms=4,
            label="MOC wall exit")
    ax.axhline(0.0, color=C.COLORS["muted"], lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xticks(r)
    ax.set_xticklabels([f"{v:g}" for v in r])
    ax.minorticks_off()
    ax.set_xlabel("p$_0$ / p$_a$ (–)")
    ax.set_ylabel("Schmucker p/p$_{sep}$ − 1.2 (–)")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=6.5)
    C.panel_label(ax, "c")
    return [C.save_figure(fig, run.out_dir, "e04_q1d_vs_moc")]


if __name__ == "__main__":
    raise SystemExit(main())
