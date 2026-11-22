"""E03: local-gamma substitution vs the frozen thermally perfect Prandtl-Meyer integral.

nu_tp = int sqrt(M^2-1) dV/V along the frozen isentrope (anderson Eq. 4.35, PDF 186),
cross-checked with nu = -int sin(2 mu)/(2 gamma p) dp (ames1953 Eq. 200, PDF 19).
Opt-in diagnostic; changes nothing in raosim/.

    .venv-jax/bin/python tests/experiments/nozzle/moc_ext_e03_local_gamma.py
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
OUT = ROOT / "builds/moc_extensions_plan_2026-11-21/opus_b/e03"

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from lrekit.thermochemistry.frozen_flow import FrozenIdealGasTable, solve_frozen_nozzle_expansion  # noqa: E402
from lrekit.core.gas_dynamics import (  # noqa: E402
    isentropic_pressure_ratio, mach_from_prandtl_meyer, prandtl_meyer,
)

INK, TEAL, MUTED, ACCENT = "#19333C", "#176D76", "#51616B", "#C26A2E"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": INK, "pdf.fonttype": 42,
})

PC, TC = 8.0e6, 3500.0
MW = 0.022
T_NODES = (200.0, 500.0, 1000.0, 2000.0, 3000.0, 3800.0)
CP_A = (1250.0, 1375.0, 1510.0, 1770.0, 1990.0, 2130.0)
CP_B = tuple(CP_A[0] + 2.0 * (c - CP_A[0]) for c in CP_A)
R_GAS = 8.31446261815324 / MW
EPS_MAX = 100.0
N_PANELS = 128
N_HISTORY = (16, 32, 64, 128, 256)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raosim_hashes() -> dict:
    out = {}
    for name, mod in sorted(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if name.startswith("lrekit") and f and f.endswith(".py"):
            out[str(Path(f).resolve().relative_to(ROOT))] = sha256(Path(f))
    return out


def make_gas(label: str, cps) -> FrozenIdealGasTable:
    src = f"manufactured {label} (moc_ext E03; not a propellant)"
    return FrozenIdealGasTable(
        molecular_weight_kg_mol=MW, composition_mass_fractions={"H2O": 0.4, "CO2": 0.35, "CO": 0.25},
        temperature_nodes_k=T_NODES, cp_nodes_j_kg_k=tuple(float(c) for c in cps), source=src,
        freeze_basis="manufactured_composition", composition_state_pressure_pa=None,
        composition_state_temperature_k=None, mixture_ratio=None, generator="moc_ext_e03",
        generator_version="1", thermo_database="manufactured_analytic_cp",
        source_artifact_sha256=hashlib.sha256(src.encode()).hexdigest())


class Isentrope:
    def __init__(self, gas: FrozenIdealGasTable):
        self.g = gas
        self.R = gas.gas_constant_j_kg_k

    def V(self, T):
        return math.sqrt(max(2.0 * self.g.enthalpy_change(T, TC), 0.0))

    def M(self, T):
        return self.V(T) / math.sqrt(self.g.gamma(T) * self.R * T)

    def lnV(self, T):
        return math.log(self.V(T))

    def lnp(self, T):
        return self.g.standard_entropy_change(TC, T) / self.R

    def T_star(self):
        f = lambda T: 2.0 * self.g.enthalpy_change(T, TC) - self.g.gamma(T) * self.R * T  # noqa: E731
        return brentq(f, self.g.minimum_temperature_k, TC - 1e-9, xtol=1e-13, rtol=8.9e-16, maxiter=500)

    def T_of(self, fun, target, Ta, Tb):
        return brentq(lambda T: fun(T) - target, Tb, Ta, xtol=1e-13, rtol=8.9e-16, maxiter=500)


def simpson(y, h):
    return h / 3.0 * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-1:2].sum())


def _segment(iso, which, Ta, Tb, first, n):
    if which == "lnV":
        fun = iso.lnV

        def integrand(T):
            return math.sqrt(max(iso.M(T) ** 2 - 1.0, 0.0))
    else:
        fun = iso.lnp

        def integrand(T):
            M = iso.M(T)
            s2mu = 2.0 * math.sqrt(max(M * M - 1.0, 0.0)) / (M * M)
            return s2mu / (2.0 * iso.g.gamma(T))
    ua, ub = fun(Ta), fun(Tb)
    sgn = 1.0 if which == "lnV" else -1.0
    if first:
        tau_b = math.sqrt(abs(ub - ua))
        tau = np.linspace(0.0, tau_b, n + 1)
        y = np.empty_like(tau)
        for k, t in enumerate(tau):
            u = ua + math.copysign(t * t, ub - ua)
            T = Ta if k == 0 else (Tb if k == n else iso.T_of(fun, u, Ta, Tb))
            y[k] = integrand(T) * 2.0 * t
        return simpson(y, tau_b / n)
    u = np.linspace(ua, ub, n + 1)
    y = np.empty_like(u)
    for k, uk in enumerate(u):
        T = Ta if k == 0 else (Tb if k == n else iso.T_of(fun, uk, Ta, Tb))
        y[k] = integrand(T)
    return sgn * simpson(y, (ub - ua) / n)


def nu_tp_cumulative(iso, Ts, which, n=N_PANELS, substitute=True):
    Tstar = iso.T_star()
    nodes = [t for t in T_NODES if min(Ts) < t < Tstar]
    bps = sorted(set([Tstar] + nodes + list(Ts)), reverse=True)
    acc, cum = 0.0, {Tstar: 0.0}
    for a, b in zip(bps, bps[1:]):
        acc += _segment(iso, which, a, b, first=(substitute and a == Tstar), n=n)
        cum[b] = acc
    return np.array([cum[T] for T in Ts]), Tstar


def run_gas(label, cps, area_ratios):
    gas = make_gas(label, cps)
    sol = solve_frozen_nozzle_expansion(gas, chamber_pressure_pa=PC, chamber_temperature_k=TC,
                                        expansion_ratio=EPS_MAX)
    st = [sol.station(float(e), supersonic=True) for e in area_ratios]
    iso = Isentrope(gas)
    Ts = [s.temperature_k for s in st]
    Mf = np.array([s.mach for s in st])
    nu_v, Tstar = nu_tp_cumulative(iso, Ts, "lnV")
    nu_v2, _ = nu_tp_cumulative(iso, Ts, "lnV", n=2 * N_PANELS)
    nu_p, _ = nu_tp_cumulative(iso, Ts, "lnp")
    hist = {n: nu_tp_cumulative(iso, Ts, "lnV", n=n)[0] for n in N_HISTORY if n not in (N_PANELS, 2 * N_PANELS)}
    hist[N_PANELS], hist[2 * N_PANELS] = nu_v, nu_v2
    pr = np.array([s.pressure_ratio for s in st])
    g_loc = np.array([s.gamma for s in st])
    gammas = {"local": g_loc, "chamber": gas.gamma(TC), "throat": sol.throat.gamma,
              "exit": sol.exit.gamma}
    approx = {}
    for k, g in gammas.items():
        gv = np.broadcast_to(np.asarray(g, dtype=float), Mf.shape)
        nu_pm = np.array([prandtl_meyer(m, gg) for m, gg in zip(Mf, gv)])
        p_pm = np.array([isentropic_pressure_ratio(m, gg) for m, gg in zip(Mf, gv)])
        M_inv = np.array([mach_from_prandtl_meyer(v, gg) for v, gg in zip(nu_v, gv)])
        approx[k] = {
            "dnu_deg": np.degrees(nu_pm - nu_v),
            "rel_dp": (p_pm - pr) / pr,
            "dmu_deg": np.degrees(np.arcsin(1.0 / M_inv) - np.arcsin(1.0 / Mf)),
            "dM_inverse": M_inv - Mf,
            "gamma": float(np.mean(gv)) if k != "local" else None,
        }
    return {"label": label, "cp_nodes": list(cps), "solution": sol, "area_ratio": np.array(area_ratios),
            "T": np.array(Ts), "M": Mf, "gamma_local": g_loc, "p_ratio": pr, "nu_tp": nu_v,
            "nu_tp_2n": nu_v2, "nu_ames200": nu_p, "nu_hist": hist, "T_star_own": Tstar, "approx": approx}


def stats(r, lo=1.05, hi=4.0):
    m = (r["M"] >= lo) & (r["M"] <= hi)
    out = {}
    for k, a in r["approx"].items():
        out[k] = {"gamma_used": a["gamma"],
                  "max_abs_dnu_deg_M1.05-4": float(np.max(np.abs(a["dnu_deg"][m]))),
                  "max_abs_rel_dp_M1.05-4": float(np.max(np.abs(a["rel_dp"][m]))),
                  "max_abs_dmu_deg_M1.05-4": float(np.max(np.abs(a["dmu_deg"][m]))),
                  "max_abs_dM_inverse_M1.05-4": float(np.max(np.abs(a["dM_inverse"][m]))),
                  "max_abs_dnu_deg_full": float(np.max(np.abs(a["dnu_deg"]))),
                  "max_abs_rel_dp_full": float(np.max(np.abs(a["rel_dp"])))}
    return out


def save(fig, stem):
    paths = []
    for ext in ("pdf", "png"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(str(p.relative_to(ROOT)))
    plt.close(fig)
    return paths


def figure(var):
    cols = {"local": ACCENT, "chamber": INK, "throat": TEAL, "exit": MUTED}
    names = {"local": r"$\gamma(T_{\rm local})$", "chamber": r"$\gamma_{\rm chamber}$",
             "throat": r"$\gamma_{\rm throat}$", "exit": r"$\gamma_{\rm exit}$ ($\epsilon$=100)"}
    ls = {"A": "-", "B": "--"}
    fig, axs = plt.subplots(1, 3, figsize=(6.5, 2.45), constrained_layout=True)
    for tag, r in var.items():
        m = (r["M"] >= 1.05) & (r["M"] <= 4.0)
        for k, a in r["approx"].items():
            axs[0].plot(r["M"][m], a["dnu_deg"][m], ls[tag], color=cols[k], lw=1.1,
                        label=names[k] if tag == "A" else None)
            axs[1].plot(r["M"][m], 100.0 * a["rel_dp"][m], ls[tag], color=cols[k], lw=1.1)
        axs[2].semilogx(r["area_ratio"], r["gamma_local"], ls[tag], color=INK, lw=1.1,
                        label=f"table {tag}")
    axs[0].set_xlabel("frozen Mach number $M_f$ (-)")
    axs[0].set_ylabel(r"$\nu_{PM}(M_f;\gamma) - \nu_{tp}$ (deg)")
    axs[1].set_xlabel("frozen Mach number $M_f$ (-)")
    axs[1].set_ylabel(r"relative error in $p/p_0$ (%)")
    axs[2].set_xlabel(r"area ratio $A/A^*$ (-)")
    axs[2].set_ylabel(r"$\gamma(T)$ along the isentrope (-)")
    for ax in axs:
        ax.axhline(0, color=MUTED, lw=0.5) if ax is not axs[2] else None
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    h0, l0 = axs[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    h0 += [Line2D([], [], color=MUTED, ls="-"), Line2D([], [], color=MUTED, ls="--")]
    l0 += ["table A (manufactured)", "table B (2x cp slope)"]
    fig.legend(h0, l0, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.0),
               fontsize=7, handlelength=1.8, columnspacing=1.0)
    return save(fig, "e03_local_gamma_error")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    eps = np.unique(np.concatenate([1.0 + np.geomspace(1e-4, 1e-2, 12),
                                    np.geomspace(1.0101, EPS_MAX, 260)]))
    verif = {}
    for gam in (1.2, 1.4):
        cp = gam * R_GAS / (gam - 1.0)
        r = run_gas(f"constant cp gamma={gam}", [cp] * len(T_NODES), eps)
        nu_pm = np.array([prandtl_meyer(m, gam) for m in r["M"]])
        p_pm = np.array([isentropic_pressure_ratio(m, gam) for m in r["M"]])
        plain, _ = nu_tp_cumulative(Isentrope(r["solution"].gas), list(r["T"]), "lnV", substitute=False)
        verif[str(gam)] = {
            "cp_j_kg_k": cp, "M_range": [float(r["M"].min()), float(r["M"].max())],
            "max_abs_nu_tp_minus_nu_PM_deg": float(np.max(np.abs(np.degrees(r["nu_tp"] - nu_pm)))),
            "max_abs_ames200_minus_nu_PM_deg": float(np.max(np.abs(np.degrees(r["nu_ames200"] - nu_pm)))),
            "max_abs_quadrature_n_vs_2n_deg": float(np.max(np.abs(np.degrees(r["nu_tp"] - r["nu_tp_2n"])))),
            "max_rel_p_ratio_diff": float(np.max(np.abs(p_pm - r["p_ratio"]) / r["p_ratio"])),
            "plain_simpson_no_substitution_max_err_deg": float(np.max(np.abs(np.degrees(plain - nu_pm)))),
        }
        verif[str(gam)]["panel_history_max_abs_err_vs_nu_PM_deg"] = {
            str(n): float(np.max(np.abs(np.degrees(v - nu_pm)))) for n, v in sorted(r["nu_hist"].items())}
        verif[str(gam)]["pass"] = (verif[str(gam)]["max_abs_nu_tp_minus_nu_PM_deg"] <= 1e-9
                                   and verif[str(gam)]["max_rel_p_ratio_diff"] <= 1e-10)
    verified = all(v["pass"] for v in verif.values())
    var, comp = {}, {}
    if verified:
        for tag, cps in (("A", CP_A), ("B", CP_B)):
            r = run_gas(f"table {tag}", cps, eps)
            var[tag] = r
            comp[tag] = {
                "cp_nodes_j_kg_k": list(cps), "T_star_K": r["T_star_own"],
                "gamma_chamber": float(r["solution"].gas.gamma(TC)),
                "gamma_throat": float(r["solution"].throat.gamma),
                "gamma_exit_eps100": float(r["solution"].exit.gamma),
                "M_exit_eps100": float(r["solution"].exit.mach),
                "nu_tp_exit_deg": float(np.degrees(r["nu_tp"][-1])),
                "max_abs_ames200_minus_nu_tp_deg": float(np.max(np.abs(np.degrees(r["nu_ames200"] - r["nu_tp"])))),
                "max_abs_quadrature_n_vs_2n_deg": float(np.max(np.abs(np.degrees(r["nu_tp"] - r["nu_tp_2n"])))),
                "approximations": stats(r),
            }
        figs = figure(var)
        np.savez(OUT / "e03_curves.npz", **{f"{t}_{k}": v for t, r in var.items()
                                             for k, v in (("area_ratio", r["area_ratio"]), ("M", r["M"]),
                                                          ("T", r["T"]), ("gamma", r["gamma_local"]),
                                                          ("nu_tp", r["nu_tp"]), ("p_ratio", r["p_ratio"]))})
    else:
        figs = []
    status = "PASS" if verified else "FAIL"
    worst = {t: max(c["approximations"].items(), key=lambda kv: kv[1]["max_abs_dnu_deg_M1.05-4"])
             for t, c in comp.items()}
    hyp = "undetermined"
    if comp:
        mx = max(a["max_abs_dnu_deg_M1.05-4"] for c in comp.values() for a in c["approximations"].values())
        mn = min(a["max_abs_dnu_deg_M1.05-4"] for c in comp.values() for a in c["approximations"].values())
        hyp = (f"supported in kind, not in size: max |dnu| over M 1.05-4 is {mn:.2f}-{mx:.2f} deg across "
               "approximations and tables, i.e. several degrees rather than 'order a degree', for this "
               "manufactured gas (gamma rises from about 1.22 to 1.33 between the chamber and M 4); "
               "the exact constant-cp reduction is verified")
    result = {
        "id": "E03", "status": status,
        "evidence_class": "verification (constant-cp limit); neither for the approximation comparison "
                          "(manufactured gas, not a propellant)",
        "configuration": {
            "Pc_Pa": PC, "Tc_K": TC, "MW_kg_mol": MW, "T_nodes_K": list(T_NODES),
            "table_A_cp": list(CP_A),
            "table_B_cp": list(CP_B),
            "table_B_rule": "cp_B(T_i) = cp_A(200 K) + 2 (cp_A(T_i) - cp_A(200 K)): every segment slope "
                            "doubled, pivot at the lowest node (spec says only 'doubled cp slope')",
            "constant_cp_tables": "cp = gamma R/(gamma-1) at gamma 1.2 and 1.4, same R",
            "area_ratio_grid": {"n": int(eps.size), "min": float(eps.min()), "max": float(eps.max())},
            "nu_tp": "anderson Eq. 4.35 (PDF 186 / printed 170): nu = int sqrt(M^2-1) d(ln V) from M = 1; "
                     "composite Simpson in ln V, breakpoints at cp nodes and stations, "
                     f"{N_PANELS} panels per sub-interval, substitution ln V - ln V* = t^2 on the first "
                     "sub-interval to remove the sqrt endpoint singularity",
            "cross_check": "ames1953 (NACA TR 1135) Eq. 200 (PDF 19 / printed 630): "
                           "nu = -int sin(2 mu)/(2 gamma p) dp, Simpson in ln p",
            "approximations": "nu_PM(M_f; gamma_x) with M_f the frozen Mach number and gamma_x local, "
                              "chamber (T = Tc), throat or exit (eps = 100)",
            "mach_angle_error_definition": "mu(M_x) - mu(M_f) with M_x = inverse nu_PM(nu_tp; gamma_x), "
                                           "i.e. the Mach angle a closed-form net would assign to the true "
                                           "turning; at fixed M_f the Mach angle is gas-independent",
            "stats_window": "M_f in [1.05, 4]; full-range maxima also reported",
        },
        "command": ".venv-jax/bin/python tests/experiments/nozzle/moc_ext_e03_local_gamma.py",
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                   text=True).stdout.strip(),
        "hashes": {"script": {str(Path(__file__).resolve().relative_to(ROOT)): sha256(Path(__file__))},
                   "raosim_imported": raosim_hashes(),
                   "inputs": {"tests/test_frozen_flow.py (table A origin)": sha256(ROOT / "tests/test_frozen_flow.py")}},
        "metrics": {"verification_constant_cp": verif, "comparison": comp},
        "hypotheses": {"H_E03": hyp,
                       "worst_approximation_per_table": {t: w[0] for t, w in worst.items()}},
        "figures": [{"paths": figs, "caption_facts": {
            "configuration": "Manufactured frozen gas (MW 22 g/mol, piecewise-linear cp table A from "
                             "tests/test_frozen_flow.py and table B with doubled slope), Pc 8 MPa, Tc 3500 K, "
                             "A/A* 1-100: (a) error of the closed-form PM angle at the frozen Mach number with "
                             "local, chamber, throat or exit gamma against the thermally perfect integral; "
                             "(b) matching p/p0 error; (c) gamma(T) along the isentrope.",
            "evidence_class": "neither (manufactured gas); the integrator itself is verified against the "
                              "constant-cp closed form",
            "not_established": "Error for a real propellant mixture or for equilibrium flow; the effect on a "
                               "2-D net (needs T1)."}}],
        "wall_time_s": time.time() - t0,
        "notes": ["First run used 16 Simpson panels per sub-interval and missed the 1e-9 deg verification "
                  "bound (6.3e-9 deg); the panel history shows 4th-order convergence (16: 6.3e-9, 32: 3.9e-10, "
                  "64: 2.5e-11, 128: 1.7e-12 deg), so 128 panels are used. The acceptance threshold was not "
                  "changed.",
                  "Plain Simpson in ln V without the t^2 substitution is limited to about 4e-6 deg by the "
                  "sqrt endpoint singularity at M = 1.",
                  "The magnitude depends on the manufactured cp slope; it is not a propellant result.",
                  "Table A reaches M_f about 5.8 at A/A* = 100 (T about 590 K), so the full-range "
                  "maxima extend beyond the M 1.05-4 window named in the spec."],
    }
    for _f in result["figures"]:
        _f.update({"script": str(Path(__file__).resolve().relative_to(ROOT)), "command": result["command"],
                   "git_head": result["git_head"], "status": result["status"],
                   "wall_time_s": result["wall_time_s"], "hashes": "see result.hashes (script, raosim, inputs)",
                   "file_sha256": {q: sha256(ROOT / q) for q in _f["paths"]}})
    (OUT / "result.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"status": status, "verif": verif, "comp": comp}, indent=1, default=float))


if __name__ == "__main__":
    main()
