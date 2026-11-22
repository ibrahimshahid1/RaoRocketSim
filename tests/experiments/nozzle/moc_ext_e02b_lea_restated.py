"""E02b: DIAGNOSTIC benchmark-definition study for the LEA TOP length reference (answer A8).

Restates the LEA bell with the source's 0.5 r_t throat arc and x_E - x_N = 15.0 r_t, fits both the
current case geometry and the restated geometry against the digitised contour with the curve x
origin at the throat and at N, and lists the tests/test_nozzle_geometry.py failures. Case files
are untouched.

    .venv-jax/bin/python tests/experiments/nozzle/moc_ext_e02b_lea_restated.py
"""
from __future__ import annotations

from lrekit.paths import repository_root as _repository_root

import copy
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = _repository_root()
sys.path.insert(0, str(ROOT))
OUT = ROOT / "builds/moc_extensions_plan_2026-11-21/opus_b/e02b"
CURVE = ROOT / "raosim/benchmark_data/curves/lea_top_contour.csv"
CASE = ROOT / "raosim/benchmark_data/cases/lea_top_schomberg_2014.json"

import numpy as np  # noqa: E402

from lrekit import benchmarks as bm  # noqa: E402
from lrekit.components.nozzle.geometry import bell_nozzle_contour  # noqa: E402

INK, TEAL, MUTED, ACCENT = "#19333C", "#176D76", "#51616B", "#C26A2E"
T15 = math.tan(math.radians(15.0))
SEC15M1 = 1.0 / math.cos(math.radians(15.0)) - 1.0
TOL = 0.08


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def raosim_hashes() -> dict:
    out = {}
    for name, mod in sorted(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if name.startswith("lrekit") and f and f.endswith(".py"):
            out[str(Path(f).resolve().relative_to(ROOT))] = sha256(Path(f))
    return out


def L_sharp(eps):
    return (math.sqrt(eps) - 1.0) / T15


def L_round(eps, rd):
    return (math.sqrt(eps) - 1.0 + rd * SEC15M1) / T15


def load_curve():
    rows = [ln.split(",") for ln in CURVE.read_text().strip().splitlines()[1:]]
    return np.array([[float(a), float(b)] for a, b in rows])


def fit(contour, Rt, curve, shift_over_rt):
    x = np.asarray(contour["x"], dtype=float) / Rt
    y = np.asarray(contour["y"], dtype=float) / Rt
    xq = curve[:, 0] + shift_over_rt
    mask = (xq >= x.min()) & (xq <= x.max())
    err = np.interp(xq[mask], x, y) - curve[mask, 1]
    return {"n_points_used": int(mask.sum()), "n_points_total": int(len(curve)),
            "points_dropped_x_ref": curve[~mask, 0].tolist(),
            "rms": float(np.sqrt(np.mean(err ** 2))), "max_abs": float(np.max(np.abs(err))),
            "within_0.08_report_only": float(np.sqrt(np.mean(err ** 2))) <= TOL,
            "errors": err.tolist(), "x_ref_used": curve[mask, 0].tolist()}


def geometry_facts(contour, Rt):
    N, E = contour["N"], contour["E"]
    return {"L_over_Rt": float(contour["Ln"]) / Rt, "x_N_over_Rt": float(N[0]) / Rt, "y_N_over_Rt": float(N[1]) / Rt,
            "x_E_over_Rt": float(E[0]) / Rt, "y_E_over_Rt": float(E[1]) / Rt,
            "x_E_minus_x_N_over_Rt": (float(E[0]) - float(N[0])) / Rt,
            "theta_n_deg": float(contour["theta_n"]), "theta_e_deg": float(contour["theta_e"]),
            "Rd_over_Rt": float(contour["Rd"]) / Rt}


def pytest_failures():
    p = subprocess.run([str(ROOT / ".venv-jax/bin/python"), "-m", "pytest", "-q", "-rf", "-p", "no:cacheprovider",
                        "tests/test_nozzle_geometry.py"], cwd=ROOT, capture_output=True, text=True)
    out = p.stdout + p.stderr
    (OUT / "pytest_test_nozzle_geometry.log").write_text(out, encoding="utf-8")
    causes = {}
    for m in re.finditer(r"^_{3,} (\S+) _{3,}$\n(.*?)(?=^_{3,} |^=+ short test summary)", out, re.M | re.S):
        body = m.group(2)
        e_lines = [ln[2:].strip() for ln in body.splitlines() if ln.startswith("E ")][:2]
        if "outside the held SP-125 chart domain" in body and "various_expansion" in m.group(1):
            cause = ("chart-domain refusal: eps = 4 is below the held chart (eps 5-50) and lookup_angles now "
                     "raises instead of extrapolating; the test loops over eps 4, 10, 25, 50")
        elif "outside the held SP-125 chart domain" in body:
            cause = ("chart-domain refusal: eps = 60 is outside the held chart and lookup_angles now raises; "
                     "the test expects a flagged extrapolated contour with a warning")
        elif "c['Ln']" in body or "expected_Ln" in body:
            cause = ("length convention: asserts Ln = 0.8 x sharp-corner cone; host now uses the SP-125 "
                     "Eq. 4-7 rounded cone (0.12992 vs 0.12912 m)")
        elif "28.0 <= tn" in body:
            cause = ("chart angle table change: lookup_angles(10, 80) now returns theta_n 25.74 deg "
                     "(SP-125 Fig. 4-14 reading via rao_top_chart); the test expects 28-32 deg (old table)")
        else:
            cause = "unclassified"
        causes[m.group(1)] = {"cause": cause, "assertion": e_lines}
    summary = [ln for ln in out.splitlines() if re.search(r"\d+ (passed|failed)", ln)]
    return {"command": ".venv-jax/bin/python -m pytest -q -rf -p no:cacheprovider tests/test_nozzle_geometry.py",
            "summary": summary[-1] if summary else None,
            "failed": sorted(set(re.findall(r"^FAILED (\S+)", out, re.M))), "causes": causes,
            "log": str((OUT / "pytest_test_nozzle_geometry.log").relative_to(ROOT))}


def figure(curve, cur, res, fits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "axes.labelsize": 8.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
                         "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
                         "ytick.color": INK, "pdf.fonttype": 42})
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.6), constrained_layout=True,
                            gridspec_kw={"width_ratios": [1.5, 1.0]})
    ax = axs[0]
    for (c, Rt), col, lab in ((cur, INK, "current case (Rd 0.382, Ln 15.045)"),
                              (res, TEAL, "restated (Rd 0.5, x_E - x_N = 15)")):
        x = np.asarray(c["x"]) / Rt
        y = np.asarray(c["y"]) / Rt
        m = x >= -0.2
        ax.plot(x[m], y[m], color=col, lw=1.0, label=lab)
    ax.plot(curve[:, 0], curve[:, 1], "o", ms=3.5, mfc="white", mec=ACCENT, mew=0.9,
            label="digitised curve, x origin at throat")
    ax.plot(curve[:, 0] + res[0]["N"][0] / res[1], curve[:, 1], "s", ms=3.0, mfc="none", mec=MUTED, mew=0.7,
            label="digitised curve, x origin at N (restated)")
    ax.set_xlabel("$x/r_t$ (-)")
    ax.set_ylabel("$r/r_t$ (-)")
    ax.legend(frameon=False, loc="lower right")
    ax2 = axs[1]
    labels, rms = [], []
    for k, v in fits.items():
        labels.append(k.replace("_", "\n", 1))
        rms.append(v["rms"])
    cols = [INK, INK, TEAL, TEAL]
    hatches = ["", "////", "", "////"]
    for i, (rv, c, h) in enumerate(zip(rms, cols, hatches)):
        ax2.bar(i, rv, 0.65, color="white" if h else c, edgecolor=c, hatch=h, lw=0.9)
        ax2.text(i, rv + 0.004, f"{rv:.3f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax2.axhline(TOL, color=ACCENT, lw=0.8, ls="--")
    ax2.text(1.5, TOL + 0.003, "0.08 (report-only)", ha="center", fontsize=6.5, color=ACCENT)
    ax2.set_xticks(range(len(rms)))
    ax2.set_xticklabels(["current\nthroat", "current\nN", "restated\nthroat", "restated\nN"], fontsize=7)
    ax2.set_ylabel("contour RMS error ($r/r_t$)")
    for a in axs:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    paths = []
    for ext in ("pdf", "png"):
        p = OUT / f"e02b_lea_restated_fit.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(str(p.relative_to(ROOT)))
    plt.close(fig)
    return paths


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    case = copy.deepcopy(bm.load_benchmark_case("lea_top_schomberg_2014"))
    inp = case["inputs"]
    Rt, eps = float(inp["Rt"]), float(inp["epsilon"])
    tn, te = float(inp["theta_n"]), float(inp["theta_e"])
    cur = bm._build_contour(case, "bezier")
    rd = 0.5
    x_N = rd * math.sin(math.radians(tn))
    L_target = 15.0 + x_N
    pct_round = 100.0 * L_target / L_round(eps, rd)
    pct_sharp = 100.0 * L_target / L_sharp(eps)
    res = bell_nozzle_contour(Rt, eps, tn, te, pct_round, gamma=float(inp["gamma"]), Rd_factor=rd)
    curve = load_curve()
    fits = {
        "current_throat_origin": fit(cur, Rt, curve, 0.0),
        "current_N_origin": fit(cur, Rt, curve, float(cur["N"][0]) / Rt),
        "restated_throat_origin": fit(res, Rt, curve, 0.0),
        "restated_N_origin": fit(res, Rt, curve, float(res["N"][0]) / Rt),
    }
    builtin = next(m for m in bm.compare_contour_to_reference(cur, case, method="bezier")
                   if m["name"] == "contour_rms_error")
    y_m = 1.0 + rd * (1.0 - math.cos(math.radians(tn)))
    tests = pytest_failures()
    figs = figure(curve, (cur, Rt), (res, Rt), fits)
    best = min(fits, key=lambda k: fits[k]["rms"])
    result = {
        "id": "E02b", "status": "PASS-DIAGNOSTIC",
        "labels": ["DIAGNOSTIC", "benchmark-definition study; case files untouched"],
        "evidence_class": "neither (benchmark-definition diagnostic); the contour comparison is report-only "
                          "benchmark agreement against a digitised figure",
        "from_answer": "A8 (opus_a/answers_round1.json)",
        "configuration": {
            "restated_geometry": {"Rd_over_rt": rd, "theta_N_deg": tn, "theta_E_deg": te,
                                  "y_E_over_rt": math.sqrt(eps), "x_E_minus_x_N_over_rt": 15.0,
                                  "basis": "schomberg2019 Fig. 5 (PDF 5: throat arc 0.5 y_t; x_mn = x_n - x_m, an "
                                           "inference) and Table 3 (PDF 6)",
                                  "construction": "lrekit.components.nozzle.geometry.bell_nozzle_contour(Rt, eps, 34, 4, "
                                                  "length_pct, Rd_factor=0.5) with length_pct chosen on the "
                                                  "rounded basis (Rd 0.5) so that Ln = 15 + 0.5 sin 34 deg"},
            "y_mn_ambiguity": {"read_as": "y_n = y_E = 5.51 r_t (used)",
                               "alternative": "y_mn = y_n - y_m would give y_E = 5.51 + y_m = "
                                              f"{5.51 + y_m:.4f} r_t (eps {((5.51 + y_m) ** 2):.2f}); not built"},
            "curve": str(CURVE.relative_to(ROOT)),
            "alignments": {"throat": "curve x measured from the throat (as compare_contour_to_reference)",
                           "N": "curve x measured from N: x_host = x_ref + x_N of that geometry"},
            "fit_rule": "linear interpolation of the host contour at the curve x; curve points outside the host x "
                        "range are dropped (same masking as compare_contour_to_reference); tolerance 0.08 "
                        "unchanged, report-only",
        },
        "command": ".venv-jax/bin/python tests/experiments/nozzle/moc_ext_e02b_lea_restated.py",
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                   text=True).stdout.strip(),
        "hashes": {"script": {str(Path(__file__).resolve().relative_to(ROOT)): sha256(Path(__file__))},
                   "raosim_imported": raosim_hashes(),
                   "inputs": {str(CASE.relative_to(ROOT)): sha256(CASE), str(CURVE.relative_to(ROOT)): sha256(CURVE)}},
        "metrics": {
            "restated": {**geometry_facts(res, Rt), "length_pct_rounded_basis_Rd0.5": pct_round,
                         "length_pct_sharp_basis": pct_sharp, "L_target_over_rt": L_target},
            "current": {**geometry_facts(cur, Rt), "length_pct": float(inp["length_pct"])},
            "contour_fits": {k: {kk: vv for kk, vv in v.items() if kk not in ("errors", "x_ref_used")}
                             for k, v in fits.items()},
            "contour_fit_errors_by_point": {k: {"x_ref": v["x_ref_used"], "error": v["errors"]} for k, v in fits.items()},
            "builtin_contour_rms_current": builtin["value"],
            "builtin_matches_current_throat_fit": abs(builtin["value"] - fits["current_throat_origin"]["rms"]) < 1e-12,
            "lowest_rms_alignment": best,
            "test_nozzle_geometry": tests,
        },
        "findings": [
            f"Restated bell: throat-to-exit L = {geometry_facts(res, Rt)['L_over_Rt']:.4f} r_t; length_pct "
            f"{pct_round:.4f} (rounded cone, Rd 0.5) or {pct_sharp:.4f} (sharp cone).",
            "Contour RMS (0.08 report-only): " + ", ".join(f"{k} {v['rms']:.4f}" for k, v in fits.items())
            + f"; lowest: {best}.",
            "The digitised curve starts at (0, 1.00); the restated geometry's N sits at y_N = "
            f"{geometry_facts(res, Rt)['y_N_over_Rt']:.4f} r_t, so the curve's own first point favours a throat "
            "origin, but the curve is a 12-point digitisation of a figure.",
        ],
        "figures": [{"paths": figs, "caption_facts": {
            "configuration": "LEA TOP bell (eps 30.3601, theta 34/4 deg) as currently built by the case "
                             "(Rd 0.382 r_t, L 15.045 r_t) and restated with the source's 0.5 r_t throat arc and "
                             "x_E - x_N = 15 r_t, against the 12-point digitised contour with x origin at the "
                             "throat or at N; right: contour RMS error per geometry and alignment.",
            "evidence_class": "neither (DIAGNOSTIC benchmark-definition study); report-only comparison with a "
                              "digitised figure",
            "not_established": "Which reading of x_mn the source intends (user decision D11); performance."}}],
        "wall_time_s": None,
        "notes": ["No case JSON, tolerance or convention was changed."],
    }
    result["wall_time_s"] = time.time() - t0
    for _f in result["figures"]:
        _f.update({"script": str(Path(__file__).resolve().relative_to(ROOT)), "command": result["command"],
                   "git_head": result["git_head"], "status": result["status"],
                   "wall_time_s": result["wall_time_s"], "hashes": "see result.hashes (script, raosim, inputs)",
                   "file_sha256": {q: sha256(ROOT / q) for q in _f["paths"]}})
    (OUT / "result.json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"restated": result["metrics"]["restated"], "fits": result["metrics"]["contour_fits"],
                      "builtin": builtin["value"], "tests": tests["causes"]}, indent=1, default=float))


if __name__ == "__main__":
    main()
