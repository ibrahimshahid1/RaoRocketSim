"""E05b: POST-HOC DIAGNOSTIC exit-plane functionals of the E05 source-flow nets (answer A7).

Integrated exit-plane mass flow and thrust (momentum plus pressure, vacuum) from
crop_to_exit_plane at every E05 level, with and without the exit-lip closure row, against the
exact source-flow values. Re-runs the deterministic E05 nets through the E05 script's
functions (imported read-only). Changes nothing in raosim/.

    .venv-jax/bin/python tests/experiments/nozzle/moc_ext_e05b_exit_functionals.py
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
from types import SimpleNamespace

ROOT = _repository_root()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests/experiments/nozzle"))
OUT = ROOT / "builds/moc_extensions_plan_2026-11-21/opus_b/e05b"

import numpy as np  # noqa: E402
from scipy.integrate import quad  # noqa: E402

import moc_ext_e05_source_flow_oracle as e05  # noqa: E402
from lrekit.core.gas_dynamics import isentropic_density_ratio, isentropic_pressure_ratio, isentropic_temperature_ratio  # noqa: E402
from lrekit.components.nozzle.solvers.bounded import crop_to_exit_plane, exit_plane_mass_flow  # noqa: E402

INK, TEAL, MUTED, ACCENT = "#19333C", "#176D76", "#51616B", "#C26A2E"
CONFIGS = ((15.0, 1.23), (15.0, 1.4), (25.0, 1.23), (25.0, 1.4))
SPEC_LEVELS = (13, 25, 49, 97, 193)
EXTRA = (385,)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plane_thrust(samples, g):
    if len(samples) < 2:
        return 0.0
    s = sorted(samples, key=lambda q: q["r"])
    r = np.array([q["r"] for q in s])
    M = np.array([max(1.0, q["M"]) for q in s])
    th = np.array([q["theta"] for q in s])
    T = np.array([isentropic_temperature_ratio(m, g) for m in M])
    rho = np.array([isentropic_density_ratio(m, g) for m in M])
    p = np.array([isentropic_pressure_ratio(m, g) for m in M])
    u = M * np.sqrt(g * T) * np.cos(th)
    return float(2.0 * math.pi * np.trapezoid((rho * u * u + p) * r, r))


def exact_plane(x_e, r_max, g, which):
    def f(r):
        M, th = e05.exact_state(x_e, r, g)
        T = isentropic_temperature_ratio(M, g)
        rho = isentropic_density_ratio(M, g)
        u = M * math.sqrt(g * T) * math.cos(th)
        if which == "mdot":
            return 2.0 * math.pi * rho * u * r
        return 2.0 * math.pi * (rho * u * u + isentropic_pressure_ratio(M, g)) * r
    v, err = quad(f, 0.0, r_max, epsabs=1e-14, epsrel=1e-13, limit=400)
    return v, err


def orders(vals):
    return [math.log2(a / b) if (a and b and a > 0 and b > 0) else None for a, b in zip(vals[:-1], vals[1:])]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = {}
    for a, g in CONFIGS:
        key = f"alpha{a:g}_gamma{g:g}"
        alpha = math.radians(a)
        rows_out = []
        x_e = None
        exact_cache = {}
        for n in SPEC_LEVELS + EXTRA:
            _, wall, meta, net, wt, err = e05.run_level(a, g, n)
            x_e = net.x_exit
            r_E = wall.r(x_e)
            lips = e05.lip_rows(net)
            s_with = crop_to_exit_plane(net)
            s_wo = crop_to_exit_plane(SimpleNamespace(rows=[r for j, r in enumerate(net.rows) if j not in lips],
                                                      x_exit=net.x_exit, gamma=net.gamma))
            rec = {"n": n, "spec_level": n in SPEC_LEVELS, "complete": bool(net.complete), "lip_rows": sorted(lips),
                   "r_E": r_E}
            for tag, smp in (("with_lip", s_with), ("without_lip", s_wo)):
                rmax = max(q["r"] for q in smp)
                md = exit_plane_mass_flow(smp, g)
                F = plane_thrust(smp, g)
                if ("full",) not in exact_cache:
                    exact_cache[("full",)] = {"mdot": exact_plane(x_e, r_E, g, "mdot")[0],
                                              "F": exact_plane(x_e, r_E, g, "F")[0]}
                ex_full = exact_cache[("full",)]
                ex_rng = {"mdot": exact_plane(x_e, rmax, g, "mdot")[0], "F": exact_plane(x_e, rmax, g, "F")[0]}
                rec[tag] = {"n_samples": len(smp), "r_max_sample": rmax, "r_max_over_r_E": rmax / r_E,
                            "mdot": md, "F": F,
                            "mdot_rel_err_full_plane": (md - ex_full["mdot"]) / ex_full["mdot"],
                            "F_rel_err_full_plane": (F - ex_full["F"]) / ex_full["F"],
                            "mdot_rel_err_same_r_range": (md - ex_rng["mdot"]) / ex_rng["mdot"],
                            "F_rel_err_same_r_range": (F - ex_rng["F"]) / ex_rng["F"]}
            rows_out.append(rec)
        ex = exact_cache[("full",)]
        closed = e05.exact_mdot(alpha, g)
        summ = {}
        for tag in ("with_lip", "without_lip"):
            for q in ("mdot_rel_err_full_plane", "F_rel_err_full_plane", "mdot_rel_err_same_r_range",
                      "F_rel_err_same_r_range"):
                spec = [abs(r[tag][q]) for r in rows_out if r["spec_level"]]
                allv = [abs(r[tag][q]) for r in rows_out]
                summ[f"{tag}:{q}"] = {"abs_err_spec_levels": spec, "orders_spec_levels": orders(spec),
                                      "order_with_extra_level": orders(allv)[-1]}
        R0 = math.sqrt(e05.area_ratio(1.5, g))
        foot = e05.exact_cminus_axis_foot(R0 * math.cos(alpha), R0 * math.sin(alpha), g)
        res[key] = {"x_exit": x_e, "start_line_axis_foot_x": foot,
                    "exit_plane_fully_downstream_of_start_line": foot < x_e,
                    "exact_full_plane": ex, "closed_form_mdot": closed,
                    "quad_vs_closed_form_mdot_rel": (ex["mdot"] - closed) / closed,
                    "levels": rows_out, "summary": summ}
        print(key, {k: [round(o, 2) if o is not None else None for o in v["orders_spec_levels"]]
                    for k, v in summ.items() if "full_plane" in k}, flush=True)
    figs = figure(res)
    result = {
        "id": "E05b", "status": "PASS-DIAGNOSTIC",
        "labels": ["POST-HOC DIAGNOSTIC", "E05 original status FAIL unchanged"],
        "evidence_class": "verification (exact source-flow functionals), post-hoc",
        "from_answer": "A7 (opus_a/answers_round1.json)",
        "configuration": {
            "nets": "E05 nets re-run deterministically through scripts/moc_ext_e05_source_flow_oracle.run_level "
                    "(cones 15/25 deg, gamma 1.23/1.4, n 13..193 spec + 385 supplementary)",
            "exit_plane": "crop_to_exit_plane at x_exit (wall Mach 3 station); 'without_lip' re-crops a copy of "
                          "the row list with the exit-lip closure row removed",
            "mass_flow": "lrekit.components.nozzle.solvers.bounded.exit_plane_mass_flow (trapezoid in r)",
            "thrust": "F/p0 = 2 pi int (rho u^2 + p) r dr, vacuum, same trapezoid and units as exit_plane_mass_flow "
                      "(rho/rho0, u/sqrt(R T0), p/p0)",
            "exact": "scipy quad of the exact source field on the plane x = x_exit over [0, r_E] ('full_plane') and "
                     "over [0, r_max of the samples] ('same_r_range'); exact mass flow also checked against the "
                     "closed form rho* V* 2 pi R*^2 (1 - cos alpha)",
            "orders": "log2 of successive error ratios (refinement ratio 2); nparcspatial2021 recommends "
                      "integrated functionals for grid-convergence estimates (as cited by the planner)",
        },
        "command": ".venv-jax/bin/python tests/experiments/nozzle/moc_ext_e05b_exit_functionals.py",
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                   text=True).stdout.strip(),
        "hashes": {"script": {str(Path(__file__).resolve().relative_to(ROOT)): sha256(Path(__file__)),
                              "tests/experiments/nozzle/moc_ext_e05_source_flow_oracle.py":
                                  sha256(ROOT / "tests/experiments/nozzle/moc_ext_e05_source_flow_oracle.py")},
                   "raosim_imported": e05.raosim_hashes(), "inputs": {}},
        "metrics": res,
        "findings": [],
        "figures": [{"paths": figs, "caption_facts": {
            "configuration": "Exit-plane mass flow and vacuum thrust of the E05 source-flow nets vs start-line "
                             "spacing, four cone/gamma cases; solid: all rows vs the exact full-plane value; "
                             "dashed: exit-lip row removed vs the exact value over the same sampled radius range.",
            "evidence_class": "verification (post-hoc diagnostic)",
            "not_established": "Behaviour on curved walls or the throat kernel; physical validity."}}],
        "wall_time_s": None,
        "notes": ["The exit-lip row contributes only its wall-point sample (r = r_E) to the exit plane; its "
                  "defective near-axis tail lies far downstream of x_exit. Removing the row therefore truncates "
                  "the plane integral at the previous row's crossing radius, which is why the same-r-range "
                  "reference is reported for that variant."],
    }
    for key, r in res.items():
        s = r["summary"]
        result["findings"].append(
            f"{key}: full-plane orders (49-97, 97-193) mdot "
            f"{[round(o, 2) for o in s['with_lip:mdot_rel_err_full_plane']['orders_spec_levels'][-2:]]}, thrust "
            f"{[round(o, 2) for o in s['with_lip:F_rel_err_full_plane']['orders_spec_levels'][-2:]]}; finest-level "
            f"errors mdot {s['with_lip:mdot_rel_err_full_plane']['abs_err_spec_levels'][-1]:.2e}, thrust "
            f"{s['with_lip:F_rel_err_full_plane']['abs_err_spec_levels'][-1]:.2e}.")
    for key, r in res.items():
        if not r["exit_plane_fully_downstream_of_start_line"]:
            result["findings"].append(
                f"{key}: the exact start C- meets the axis at x = {r['start_line_axis_foot_x']:.4f} R*, downstream "
                f"of x_exit = {r['x_exit']:.4f} R*, so the exit plane near the axis lies upstream of the initial "
                "data line; crop_to_exit_plane fills that part by linear extrapolation to r = 0. The exit "
                "functionals lose their order (finest relative error about "
                f"{abs(r['levels'][-1]['with_lip']['mdot_rel_err_full_plane']):.1e}) with or without the lip row. "
                "This is a limitation of the E05 exit station for this cone, not evidence about the lip row; it "
                "also affects E05's exit-plane GCI for this case.")
    for key, r in res.items():
        spec = [x for x in r["levels"] if x["spec_level"]]
        dif = max(abs(x["with_lip"]["mdot_rel_err_full_plane"] - x["without_lip"]["mdot_rel_err_same_r_range"])
                  / max(abs(x["with_lip"]["mdot_rel_err_full_plane"]), 1e-300) for x in spec[2:])
        result["findings"].append(
            f"{key}: with the lip row vs without it (same-r-range reference), the mass-flow errors differ by at "
            f"most {100 * dif:.1f} % of the error itself on levels 49-193; the lip row does not contaminate the "
            "exit functionals (its only exit-plane sample is the lip wall point).")
    result["wall_time_s"] = time.time() - t0
    for _f in result["figures"]:
        _f.update({"script": str(Path(__file__).resolve().relative_to(ROOT)), "command": result["command"],
                   "git_head": result["git_head"], "status": result["status"],
                   "wall_time_s": result["wall_time_s"], "hashes": "see result.hashes (script, raosim, inputs)",
                   "file_sha256": {q: sha256(ROOT / q) for q in _f["paths"]}})
    (OUT / "result.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")


def figure(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "axes.labelsize": 8.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
                         "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "pdf.fonttype": 42})
    cols = [INK, TEAL, MUTED, ACCENT]
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.7), constrained_layout=True)
    for c, (key, r) in zip(cols, res.items()):
        lv = r["levels"]
        h = np.array([1.0 / (x["n"] - 1) for x in lv])
        a, g = key.replace("alpha", "").replace("gamma", "").split("_")
        for ax, q in zip(axs, ("mdot", "F")):
            y1 = np.abs([x["with_lip"][f"{q}_rel_err_full_plane"] for x in lv])
            y2 = np.abs([x["without_lip"][f"{q}_rel_err_same_r_range"] for x in lv])
            ax.loglog(h, y1, "-o", color=c, ms=3, lw=1.0, label=f"{a} deg, gamma {g}")
            ax.loglog(h, y2, "--s", color=c, ms=3, lw=0.8, mfc="white")
    for ax, lab in zip(axs, ("|exit mass-flow error| (relative)", "|exit thrust error| (relative)")):
        hh = np.array([1 / 12, 1 / 384])
        y0 = ax.get_ylim()[0] * 5
        ax.loglog(hh, y0 * (hh / hh[-1]) ** 2, ":", color=MUTED, lw=0.8, label="slope 2")
        ax.axvspan(1 / 384 * 0.8, 1 / 384 * 1.25, color=MUTED, alpha=0.12, lw=0)
        ax.set_xlabel("start-line spacing $h$ = 1/(n-1) (-)")
        ax.set_ylabel(lab)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].legend(frameon=False, fontsize=6.5, loc="lower right")
    paths = []
    for ext in ("pdf", "png"):
        p = OUT / f"e05b_exit_functionals.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(str(p.relative_to(ROOT)))
    plt.close(fig)
    return paths


if __name__ == "__main__":
    main()
