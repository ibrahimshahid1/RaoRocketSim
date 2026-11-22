"""E02: LEA/TOP length-failure diagnosis and bell length-convention inventory.

Opt-in diagnostic (builds/moc_extensions_plan_2026-11-21, experiment E02).
Changes nothing in raosim/ or the benchmark case files.

    .venv-jax/bin/python tests/experiments/nozzle/moc_ext_e02_length_convention.py
"""
from __future__ import annotations

from lrekit.paths import repository_root as _repository_root

import ast
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
OUT = ROOT / "builds/moc_extensions_plan_2026-11-21/opus_b/e02"

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lrekit import benchmarks as bm  # noqa: E402
from lrekit import nozzle_geometry as ng  # noqa: E402
from lrekit import rao_variational as rv  # noqa: E402

INK, TEAL, MUTED, ACCENT = "#19333C", "#176D76", "#51616B", "#C26A2E"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": INK, "pdf.fonttype": 42,
})

T15 = math.tan(math.radians(15.0))
SEC15M1 = 1.0 / math.cos(math.radians(15.0)) - 1.0


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def raosim_hashes() -> dict:
    out = {}
    for name, mod in sorted(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if name.startswith("lrekit") and f and f.endswith(".py"):
            out[str(Path(f).resolve().relative_to(ROOT))] = sha256(Path(f))
    return out


def L_sharp(eps: float) -> float:
    return (math.sqrt(eps) - 1.0) / T15


def L_round(eps: float, rd: float) -> float:
    return (math.sqrt(eps) - 1.0 + rd * SEC15M1) / T15


# ----------------------------------------------------------------- part 1
def per_case_lengths() -> list[dict]:
    rows = []
    for cid in bm.list_benchmark_cases():
        case = bm.load_benchmark_case(cid)
        inp = case["inputs"]
        if "length_pct" not in inp:
            continue
        eps, pct, Rt = float(inp["epsilon"]), float(inp["length_pct"]), float(inp["Rt"])
        rd = float(case.get("solver_options", {}).get("rao", {}).get(
            "throat_downstream_radius_factor", 0.382))
        ref = case.get("reference", {}).get("geometry", {}).get("length_over_rt")
        sharp = pct / 100.0 * L_sharp(eps)
        rounded = pct / 100.0 * L_round(eps, 0.382)
        host = None
        try:
            c = bm._build_contour(case, "bezier")
            host = float(c["Ln"]) / Rt
        except Exception as exc:  # recorded, not hidden
            host = f"error: {exc!r}"
        rao_closed = rv._target_length(Rt, eps, pct) / Rt
        row = {
            "case_id": cid, "epsilon": eps, "length_pct": pct, "Rt_m": Rt,
            "Rd_over_Rt_assumed": 0.382, "rao_case_Rd_over_Rt": rd,
            "status_policy": case.get("status_policy"),
            "L_over_Rt_sharp": sharp, "L_over_Rt_rounded": rounded,
            "L_over_Rt_host_bezier": host,
            "L_over_Rt_rao_target_length": rao_closed,
            "length_pct_equals_100_ref_over_Lsharp": None,
            "reference": None,
        }
        if ref is not None:
            v, tol = float(ref["value"]), float(ref["tolerance"])
            row["reference"] = {"value": v, "tolerance": tol, "mode": ref.get("mode"),
                                "source_ref": ref.get("source_ref")}
            row["length_pct_equals_100_ref_over_Lsharp"] = abs(
                pct - 100.0 * v / L_sharp(eps)) < 1e-9
            row["pct_from_ref_rounded"] = 100.0 * v / L_round(eps, 0.382)
            for k in ("sharp", "rounded"):
                d = row[f"L_over_Rt_{k}"] - v
                row[f"delta_{k}"] = d
                row[f"within_tol_{k}"] = abs(d) <= tol
            row["effective_mode_bezier"] = bm._mode_for_metric(
                case, "bezier", ref.get("mode", "strict"))
            row["effective_mode_rao"] = bm._mode_for_metric(
                case, "rao", ref.get("mode", "strict"))
        rows.append(row)
    return rows


# ----------------------------------------------------------------- part 2
HELPERS = {"_full_cone_length", "_target_length", "target_length"}


def _is_15(node) -> bool:
    return isinstance(node, ast.Constant) and node.value in (15, 15.0)


def _is_rad15(a) -> bool:
    if isinstance(a, ast.Call):
        inner = a.func.attr if isinstance(a.func, ast.Attribute) else getattr(a.func, "id", "")
        return inner in ("radians", "deg2rad") and len(a.args) == 1 and _is_15(a.args[0])
    return False


def _is_tan15(node, alias15=frozenset()) -> bool:
    if isinstance(node, ast.Name) and node.id == "_TAN15":
        return True
    if not isinstance(node, ast.Call):
        return False
    fn = node.func
    name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
    if name != "tan" or not node.args:
        return False
    a = node.args[0]
    return _is_rad15(a) or (isinstance(a, ast.Name) and a.id in alias15)


def _alias15(fn_node) -> frozenset:
    names = set()
    for n in ast.walk(fn_node):
        if isinstance(n, ast.Assign) and _is_rad15(n.value):
            names.update(t.id for t in n.targets if isinstance(t, ast.Name))
    return frozenset(names)


def _stmt_parents(tree):
    parents = {}
    for node in ast.walk(tree):
        for ch in ast.iter_child_nodes(node):
            parents[ch] = node
    return parents


def _enclosing(node, parents, kinds):
    while node in parents:
        node = parents[node]
        if isinstance(node, kinds):
            return node
    return None


def _module_of(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    return ".".join(rel.parts)


def inventory() -> dict:
    files = sorted(list((ROOT / "src" / "lrekit").rglob("*.py")) + list((ROOT / "tests/experiments/nozzle").glob("*.py"))
                   + list((ROOT / "tests").rglob("*.py")))
    files = [f for f in files if not f.name.startswith("moc_ext_")]
    sites, helper_defs, calls = [], {}, []
    for f in files:
        src = f.read_text(encoding="utf-8")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        lines = src.splitlines()
        parents = _stmt_parents(tree)
        mod = _module_of(f)
        imports = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for a in node.names:
                    imports[a.asname or a.name] = (node.module, a.name)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    imports[a.asname or a.name.split(".")[0]] = (a.name, None)
        aliases = {}
        for fdef in ast.walk(tree):
            if isinstance(fdef, (ast.FunctionDef, ast.AsyncFunctionDef)):
                aliases[fdef] = _alias15(fdef)
        for node in ast.walk(tree):
            encl_fn = _enclosing(node, parents, (ast.FunctionDef, ast.AsyncFunctionDef))
            if _is_tan15(node, aliases.get(encl_fn, frozenset())):
                stmt = _enclosing(node, parents, (ast.stmt,))
                if isinstance(node, ast.Name) and isinstance(stmt, ast.Assign) and any(
                        getattr(t, "id", None) == "_TAN15" for t in stmt.targets):
                    continue
                fn = _enclosing(node, parents, (ast.FunctionDef, ast.AsyncFunctionDef))
                text = ast.get_source_segment(src, stmt) or lines[node.lineno - 1]
                if isinstance(stmt, (ast.FunctionDef, ast.If, ast.For, ast.While, ast.With)):
                    text = lines[node.lineno - 1].strip()
                text = " ".join(text.split())
                rounded = bool(re.search(r"\bcos\s*\(|sec", text))
                lengthlike = bool(re.search(r"Re\s*-\s*Rt|sqrt\(.*eps|Re - 0\.0|expected_re|- cfg\.Rt|"
                                            r"- Rt\)|length_pct|0\.8", text))
                role = "bell_reference_cone_length" if lengthlike else "other_tan15_use"
                if isinstance(stmt, ast.Assign) and any(
                        getattr(t, "id", None) == "_TAN15" for t in stmt.targets):
                    role = "constant_definition_TAN15"
                if f.name == "test_conical.py":
                    role = "actual_conical_nozzle_length (not a bell length-percent convention)"
                sites.append({
                    "file": str(f.relative_to(ROOT)), "line": node.lineno,
                    "function": fn.name if fn else None,
                    "expression": text[:240],
                    "convention": "rounded_throat_SP125_Eq4-7" if rounded else "sharp_corner",
                    "role": role,
                })
                if fn is not None and fn.name in HELPERS:
                    helper_defs[(mod, fn.name)] = sites[-1]["convention"]
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in HELPERS:
                name = fn.id
                target = imports.get(name, (mod, name))
                target = (target[0], target[1] or name)
            elif isinstance(fn, ast.Attribute) and fn.attr in HELPERS:
                name = fn.attr
                base = getattr(fn.value, "id", None)
                if base == "self":
                    target = (mod, name)
                else:
                    target = (imports.get(base, (base, None))[0], name)
            else:
                continue
            encl = _enclosing(node, parents, (ast.FunctionDef, ast.AsyncFunctionDef))
            calls.append({"file": str(f.relative_to(ROOT)), "line": node.lineno,
                          "function": encl.name if encl else None,
                          "callee": f"{target[0]}.{target[1]}",
                          "_key": target})
    for c in calls:
        c["convention"] = helper_defs.get(c.pop("_key"), "unresolved")
    notes = {
        "raosim/conical.py": "conical_nozzle_length(Rt, eps, half_angle) is the length of an actual "
                             "conical nozzle (variable half-angle, not a tan(15) literal); not a bell "
                             "length-percent convention, so it is not a tan15 site.",
        "tests/test_nozzle_geometry.py:63": "asserts bell_nozzle_contour(...)['Ln'] == 0.8*sharp at "
                                            "rel 1e-6 against the host (rounded) producer.",
        "tests/test_rao_optimizer.py:35": "asserts the rao_optimizer (sharp) producer; consistent.",
        "tests/test_phase6_coupled_wall.py:328": "compares the Rao BVP (sharp) producer; consistent.",
        "test fixtures (test_moc.py, test_moc_bounded_net.py:400, test_rao_variational_moc.py:836)":
            "build their own SplineWall with a sharp length; no producer comparison.",
    }
    return {"tan15_sites": sites,
            "helper_definitions": [{"helper": f"{k[0]}.{k[1]}", "convention": v}
                                   for k, v in sorted(helper_defs.items())],
            "helper_calls": calls, "notes": notes}


# ----------------------------------------------------------------- part 3
def run_pytest(target: str) -> dict:
    t0 = time.time()
    p = subprocess.run([str(ROOT / ".venv-jax/bin/python"), "-m", "pytest", "-q", "-rf",
                        "-p", "no:cacheprovider", target],
                       cwd=ROOT, capture_output=True, text=True)
    out = p.stdout + p.stderr
    failed = sorted(set(re.findall(r"^FAILED (\S+)", out, re.M)))
    summary = [ln for ln in out.splitlines() if re.search(r"\d+ (passed|failed)", ln)]
    (OUT / f"pytest_{Path(target).stem}.log").write_text(out, encoding="utf-8")
    causes = {}
    for m in re.finditer(r"^_{3,} (\S+) _{3,}$\n(.*?)(?=^_{3,} |^=+ short test summary)", out, re.M | re.S):
        body = m.group(2)
        if "outside the held SP-125 chart domain" in body:
            causes[m.group(1)] = "chart-domain refusal (not a length-convention failure)"
        elif re.search(r"Ln|length_over_rt|L/R_t", body):
            causes[m.group(1)] = "length convention (sharp expected vs rounded produced)"
        elif "28.0 <= tn" in body:
            causes[m.group(1)] = "chart angle table change (not a length-convention failure)"
        else:
            causes[m.group(1)] = "unclassified; see log"
    return {"failure_causes": causes,"command": f".venv-jax/bin/python -m pytest -q -rf -p no:cacheprovider {target}",
            "exit_code": p.returncode, "summary": summary[-1] if summary else None,
            "failed": failed, "wall_time_s": time.time() - t0,
            "log": str((OUT / f"pytest_{Path(target).stem}.log").relative_to(ROOT))}


def baseline_metrics(case_id: str) -> dict:
    case = bm.load_benchmark_case(case_id)
    contour = bm._build_contour(case, "bezier")
    m = []
    m += bm.compare_contour_to_reference(contour, case, method="bezier")
    m += bm.compare_performance_to_reference(contour, case, method="bezier")
    m += bm._solver_metrics(contour, case, "bezier")
    m += bm._wall_pressure_metrics(contour, case, "bezier")
    return {"overall_status": bm._overall_status(m), "all": m,
            "failing_metrics": [x["name"] for x in m if x["status"] == "fail"],
            "length_metric": next(x for x in m if x["name"] == "length_over_rt")}


# ----------------------------------------------------------------- part 4
def diagnostic_rerun(case_id: str) -> dict:
    case = copy.deepcopy(bm.load_benchmark_case(case_id))
    inp = case["inputs"]
    eps, Rt = float(inp["epsilon"]), float(inp["Rt"])
    ref = float(case["reference"]["geometry"]["length_over_rt"]["value"])
    pct_new = 100.0 * ref / L_round(eps, 0.382)
    inp["length_pct"] = pct_new
    contour = bm._build_contour(case, "bezier")
    metrics = bm.compare_contour_to_reference(contour, case, method="bezier")
    geo = [m for m in metrics if m["category"] == "geometry"]
    numeric = [m for m in geo if m["name"] != "contour_rms_error"]
    return {
        "label": "DIAGNOSTIC (local length_pct override; case file untouched)",
        "case_id": case_id, "length_pct_original": float(
            bm.load_benchmark_case(case_id)["inputs"]["length_pct"]),
        "length_pct_diagnostic": pct_new, "L_over_Rt": float(contour["Ln"]) / Rt,
        "geometry_metrics": geo,
        "strict_or_report_numeric_all_within_tolerance": all(
            m["delta"] is not None and m["delta"] <= m["tolerance"] for m in numeric),
        "statuses": {m["name"]: m["status"] for m in geo},
    }


# ----------------------------------------------------------------- figures
def save(fig, stem: str) -> list[str]:
    paths = []
    for ext in ("pdf", "png"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(str(p.relative_to(ROOT)))
    plt.close(fig)
    return paths


def lea_readings() -> dict:
    case = bm.load_benchmark_case("lea_top_schomberg_2014")
    inp = case["inputs"]
    Rt, eps, pct = float(inp["Rt"]), float(inp["epsilon"]), float(inp["length_pct"])
    ref = case["reference"]["geometry"]["length_over_rt"]
    v, tol = float(ref["value"]), float(ref["tolerance"])
    c = bm._build_contour(case, "bezier")
    xN = (float(c["N"][0]) - float(c.get("throat_location", 0.0))) / Rt
    xE = (float(c["E"][0]) - float(c.get("throat_location", 0.0))) / Rt
    tm = math.radians(float(inp["theta_n"]))
    sharp = pct / 100.0 * L_sharp(eps)
    rounded = float(c["Ln"]) / Rt
    xm_host = 0.382 * math.sin(tm)
    xm_paper = 0.5 * math.sin(tm)
    readings = {
        "i_sharp_throat_to_exit": {"value": sharp, "delta": sharp - v},
        "ii_rounded_SP125_Eq4-7_throat_to_exit_host": {"value": rounded, "delta": rounded - v},
        "iii_xmn_from_end_of_throat_arc_host_rounded": {
            "value": xE - xN, "delta": xE - xN - v,
            "x_N_over_Rt": xN, "x_E_over_Rt": xE, "host_throat_arc_Rd_over_Rt": 0.382},
        "iii_xmn_from_end_of_throat_arc_host_sharp_length": {
            "value": sharp - xm_host, "delta": sharp - xm_host - v},
    }
    for r in readings.values():
        r["within_tol"] = abs(r["delta"]) <= tol
    implied_total = v + xm_paper
    return {
        "reference": {"value": v, "tolerance": tol, "source_ref": ref.get("source_ref")},
        "readings": readings,
        "paper_reading_throat_to_exit_if_xmn_is_15": {
            "throat_arc_radius_over_Rt": 0.5, "x_m_over_Rt": xm_paper,
            "L_over_Rt": implied_total,
            "length_pct_sharp_basis": 100.0 * implied_total / L_sharp(eps),
            "length_pct_rounded_basis_Rd0.382": 100.0 * implied_total / L_round(eps, 0.382),
            "length_pct_rounded_basis_Rd0.5": 100.0 * implied_total / L_round(eps, 0.5)},
        "benchmark_expectation_matches": [k for k, r in readings.items() if r["within_tol"]],
        "basis": "schomberg2019 Table 3 (p.6) x_mn = 15.0 r_t; x_mn = x_n - x_m per Fig. 5 (p.5) is an "
                 "inference (verifier-marked); throat expansion arc r_tm = 0.5 y_t (Fig. 5).",
    }


def fig_bars(rows, lea_r=None) -> list[str]:
    rr = [r for r in rows if r["reference"] is not None]
    labels = {"lea_top_schomberg_2014": "LEA TOP\n(Schomberg et al.)",
              "vulcain_s1_separation_similarity": "Vulcain S1",
              "rao1958_nozzle_b": "Rao 1958\nNozzle B"}
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    w = 0.27
    for i, r in enumerate(rr):
        tol = r["reference"]["tolerance"]
        ax.add_patch(plt.Rectangle((i - 0.45, -tol), 0.9, 2 * tol, color=MUTED, alpha=0.18, lw=0))
        ax.bar(i - w, r["delta_sharp"], w, color=TEAL, edgecolor=TEAL, lw=1.2,
               label="sharp-corner cone" if i == 0 else None)
        ax.bar(i, r["delta_rounded"], w, color=ACCENT,
               label="rounded-throat cone, $R_d/R_t$ = 0.382" if i == 0 else None)
        for dx, v in ((-w, r["delta_sharp"]), (0.0, r["delta_rounded"])):
            ax.text(i + dx, v + 0.002, f"{v:+.4f}", ha="center", va="bottom", fontsize=7, color=INK)
        ax.text(i - 0.43, -tol - 0.002, f"tol ±{tol:g}", ha="left", va="top", fontsize=7, color=MUTED)
        if lea_r is not None and r["case_id"] == "lea_top_schomberg_2014":
            v3 = lea_r["readings"]["iii_xmn_from_end_of_throat_arc_host_rounded"]["delta"]
            ax.bar(i + w, v3, w, color="white", edgecolor=INK, hatch="////", lw=0.8,
                   label=r"$x_E - x_N$ (from end of throat arc), host bell")
            ax.text(i + w, v3 - 0.002, f"{v3:+.3f}", ha="center", va="top", fontsize=7, color=INK)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(range(len(rr)))
    ax.set_xticklabels([labels.get(r["case_id"], r["case_id"]) for r in rr])
    ax.set_ylabel(r"$L/R_t$ minus reference (-)")
    lo = -0.03
    if lea_r is not None:
        lo = min(lo, lea_r["readings"]["iii_xmn_from_end_of_throat_arc_host_rounded"]["delta"] - 0.02)
    ax.set_ylim(lo, 0.06)
    ax.legend(frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0), fontsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return save(fig, "e02a_length_delta_by_case")


def fig_map(rows) -> tuple[list[str], dict]:
    eps = np.linspace(5, 50, 181)
    pct = np.linspace(60, 100, 161)
    E, P = np.meshgrid(eps, pct)
    dL = P / 100.0 * 0.382 * SEC15M1 / T15
    rel = 100.0 * dL / (P / 100.0 * (np.sqrt(E) - 1.0) / T15)
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.7), sharey=True)
    cs = axs[0].contourf(E, P, dL, levels=12, cmap="cividis")
    cl = axs[0].contour(E, P, dL, levels=[0.035, 0.04, 0.045, 0.05], colors="w", linewidths=0.6)
    axs[0].clabel(cl, fmt="%.3f", fontsize=6.5)
    fig.colorbar(cs, ax=axs[0], label=r"$\Delta L/R_t$ (-)")
    cs2 = axs[1].contourf(E, P, rel, levels=12, cmap="cividis")
    cl2 = axs[1].contour(E, P, rel, levels=[0.3, 0.5, 1.0], colors="w", linewidths=0.6)
    axs[1].clabel(cl2, fmt="%.1f%%", fontsize=6.5)
    fig.colorbar(cs2, ax=axs[1], label=r"$\Delta L / L_{\rm sharp}$ (%)")
    for r in rows:
        if r["reference"] is None:
            continue
        for ax in axs:
            inside = 5 <= r["epsilon"] <= 50
            ax.plot(r["epsilon"], r["length_pct"], "o", ms=4, mfc=ACCENT if inside else "none",
                    mec=ACCENT, mew=1.0, clip_on=False)
    axs[0].set_ylabel("length percent of 15° cone (%)")
    for ax in axs:
        ax.set_xlim(4.5, 50)
        ax.set_xlabel(r"area ratio $\epsilon$ (-)")
    p = save(fig, "e02b_length_difference_map")
    facts = {"dL_over_Rt_min": float(dL.min()), "dL_over_Rt_max": float(dL.max()),
             "rel_pct_min": float(rel.min()), "rel_pct_max": float(rel.max()),
             "dL_independent_of_epsilon": True}
    return p, facts


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = per_case_lengths()
    lea = next(r for r in rows if r["case_id"] == "lea_top_schomberg_2014")
    repro = {
        "L_over_Rt_rounded": lea["L_over_Rt_rounded"],
        "L_over_Rt_sharp": lea["L_over_Rt_sharp"],
        "L_over_Rt_host_bezier": lea["L_over_Rt_host_bezier"],
        "abs_err_rounded_vs_15.044818739": abs(lea["L_over_Rt_rounded"] - 15.044818739),
        "abs_err_sharp_vs_15": abs(lea["L_over_Rt_sharp"] - 15.0),
        "abs_err_host_vs_rounded": abs(lea["L_over_Rt_host_bezier"] - lea["L_over_Rt_rounded"]),
    }
    repro["pass_1e-9"] = (repro["abs_err_rounded_vs_15.044818739"] <= 1e-9
                          and repro["abs_err_sharp_vs_15"] <= 1e-9
                          and repro["abs_err_host_vs_rounded"] <= 1e-9)
    inv = inventory()
    py_bench = run_pytest("tests/test_benchmarks.py")
    lea_base = baseline_metrics("lea_top_schomberg_2014")
    lea_base_all = lea_base.pop("all")
    if lea_base["failing_metrics"] == ["length_over_rt"]:
        for k in py_bench["failure_causes"]:
            py_bench["failure_causes"][k] = (
                "LEA/TOP run_benchmark overall_status 'fail'; the only failing metric is "
                "length_over_rt (length convention)")
    py_geom = run_pytest("tests/test_nozzle_geometry.py")
    diag = [diagnostic_rerun("lea_top_schomberg_2014"),
            diagnostic_rerun("vulcain_s1_separation_similarity")]
    lea_r = lea_readings()
    figs_a = fig_bars(rows, lea_r)
    figs_b, map_facts = fig_map(rows)

    known = {"tests/test_benchmarks.py::test_lea_top_bezier_benchmark_passes_explicit_geometry",
             "tests/test_benchmarks.py::test_benchmark_cli_writes_reports"}
    only_known = set(py_bench["failed"]) <= known and len(py_bench["failed"]) > 0
    findings = []
    vul = next(r for r in rows if r["case_id"] == "vulcain_s1_separation_similarity")
    findings.append({
        "id": "E02-F1", "text": (
            f"Vulcain S1: strict reference L/Rt {vul['reference']['value']:.4f} (tol "
            f"{vul['reference']['tolerance']}); host bezier gives {vul['L_over_Rt_host_bezier']:.4f} "
            f"(delta {vul['delta_rounded']:+.4f}); sharp gives {vul['L_over_Rt_sharp']:.4f}. The case "
            f"status_policy for bezier is 'report' (effective mode {vul['effective_mode_bezier']}), "
            "so no test fails. Recorded, not fixed.")})
    nb = next(r for r in rows if r["case_id"] == "rao1958_nozzle_b")
    findings.append({
        "id": "E02-F2", "text": (
            f"Rao 1958 Nozzle B: strict for method rao (sharp, {nb['L_over_Rt_rao_target_length']:.6f} vs "
            f"{nb['reference']['value']}); the rounded convention would give {nb['L_over_Rt_rounded']:.6f} "
            f"(delta {nb['delta_rounded']:+.4f} vs tol {nb['reference']['tolerance']}); the host bezier "
            f"path itself refuses this case: {nb['L_over_Rt_host_bezier']}")})
    if py_geom["failed"]:
        findings.append({"id": "E02-F3", "text": (
            "tests/test_nozzle_geometry.py (outside the benchmark file) has failures: "
            + "; ".join(f"{k}: {v}" for k, v in py_geom["failure_causes"].items())
            + ". Only test_bell_length is a length-convention failure (it asserts the sharp-corner "
            "Ln against the host rounded producer at rel 1e-6). Recorded, not fixed.")})
    findings.append({"id": "E02-F4", "text": (
        "DIAGNOSTIC LEA rerun with length_pct = 1500/L_round passes all strict geometry metrics; "
        "the report-only contour_rms_error rises from "
        f"{next(m['value'] for m in lea_base_all if m['name'] == 'contour_rms_error'):.5f} to "
        f"{next(m['value'] for m in diag[0]['geometry_metrics'] if m['name'] == 'contour_rms_error'):.5f} "
        "(tolerance 0.08, report mode) because the whole bell is shortened.")})
    rd = lea_r["readings"]
    findings.append({"id": "E02-F5", "text": (
        "LR2 reading check (schomberg2019): the encoded benchmark expectation (throat-to-exit Ln/Rt = 15 "
        "+/- 0.02) is matched only by " + ", ".join(lea_r["benchmark_expectation_matches"]) + ", and "
        "only because length_pct was derived from 1500/L_sharp (circular). Under the Fig. 5 reading "
        "x_mn = x_n - x_m, the host bell gives x_E - x_N = "
        f"{rd['iii_xmn_from_end_of_throat_arc_host_rounded']['value']:.4f} (rounded) and "
        f"{rd['iii_xmn_from_end_of_throat_arc_host_sharp_length']['value']:.4f} (sharp length), both "
        "outside 0.02; the paper's own geometry would put throat-to-exit at about "
        f"{lea_r['paper_reading_throat_to_exit_if_xmn_is_15']['L_over_Rt']:.4f} Rt. The host also uses "
        "a 0.382 Rt downstream throat arc for LEA where Fig. 5 shows 0.5 y_t. The planner's mechanical "
        "root cause stands, but the benchmark's length reference may not be a throat-to-exit length; "
        "which reading is right is not established here.")})
    findings.append({"id": "E02-F6", "text": (
        "LR1 (huzel1967): Eq. 4-7 is on PDF 99 / printed 90 and the chart L_f basis (PDF 100 / printed 91) "
        "is the rounded-throat cone; the docstring of raosim/nozzle_geometry.py:_full_cone_length cites "
        "PDF p.100 and p.107 for the equation. Page citation difference recorded, not fixed.")})
    status = "PASS" if (repro["pass_1e-9"] and only_known) else "FAIL"

    scripts_used = [Path(__file__)]
    inputs = sorted(ROOT.glob("raosim/benchmark_data/cases/*.json"))
    result = {
        "id": "E02", "status": status,
        "evidence_class": "verification of the diagnosis; benchmark agreement for the geometry metrics",
        "configuration": {
            "Rd_over_Rt_for_rounded": 0.382, "cone_half_angle_deg": 15.0,
            "sharp": "L/Rt = (pct/100)(sqrt(eps)-1)/tan15",
            "rounded": "L/Rt = (pct/100)[(sqrt(eps)-1) + (Rd/Rt)(sec15-1)]/tan15 "
                       "(huzel1967 SP-125 Eq. 4-7, PDF 99 / printed 90; R = 0.382 Rt as in the worked "
                       "example PDF 107 / printed 98; chart L_f basis PDF 100 / printed 91)",
            "lea_reading_iii": "x_mn = x_n - x_m (schomberg2019 Fig. 5 p.5, inference); host N = end of "
                               "the 0.382 Rt downstream throat arc",
            "map_domain": {"epsilon": [5, 50], "length_pct": [60, 100]},
        },
        "command": ".venv-jax/bin/python tests/experiments/nozzle/moc_ext_e02_length_convention.py",
        "git_head": git_head(),
        "hashes": {"script": {str(p.relative_to(ROOT)): sha256(p) for p in scripts_used},
                   "raosim_imported": raosim_hashes(),
                   "inputs": {str(p.relative_to(ROOT)): sha256(p) for p in inputs}},
        "metrics": {
            "reproduction": repro, "per_case": rows, "pytest_benchmarks": py_bench,
            "lea_length_readings_LR2": lea_r,
            "lea_baseline_metrics_unmodified_case": lea_base,
            "pytest_nozzle_geometry_extra": py_geom, "diagnostic_rerun": diag,
            "length_difference_map": map_facts,
            "inventory_counts": {
                "tan15_sites": len(inv["tan15_sites"]),
                "sharp": sum(s["convention"] == "sharp_corner" for s in inv["tan15_sites"]),
                "rounded": sum(s["convention"] != "sharp_corner" for s in inv["tan15_sites"]),
                "helper_calls": len(inv["helper_calls"]),
            },
        },
        "inventory": inv,
        "hypotheses": {
            "H_LEA_root_cause": ("supported (mechanism); the benchmark reference definition itself is "
                                 "open per LR2, see E02-F5") if (repro["pass_1e-9"] and lea[
                "length_pct_equals_100_ref_over_Lsharp"]) else "not supported",
            "H_two_absolute_lengths": "supported" if any(
                s["convention"] == "sharp_corner" and s["role"] == "bell_reference_cone_length"
                and s["file"].startswith("src/lrekit/") for s in inv["tan15_sites"]) else "not supported",
        },
        "findings": findings,
        "figures": [
            {"paths": figs_a, "caption_facts": {
                "configuration": "Per benchmark case with a length reference, L/Rt from the case "
                                 "length_pct under the sharp-corner and rounded-throat (Rd/Rt = 0.382) "
                                 "15-degree reference cones, minus the source value; grey band is the "
                                 "case tolerance. The tolerance is strict only for LEA (bezier) and "
                                 "Nozzle B (rao); Vulcain S1 is report-only, and the host bezier path "
                                 "refuses Nozzle B (eps 4.973 is outside the chart), so its rounded bar "
                                 "is the closed form only. Hatched bar (LEA only): host x_E - x_N, the "
                                 "inferred schomberg2019 Fig. 5 reading of x_mn.",
                "evidence_class": "verification of the diagnosis; benchmark agreement for the geometry metric",
                "not_established": "Which convention the project should adopt (D7); Schomberg's x_mn "
                                   "definition (LR2)."}},
            {"paths": figs_b, "caption_facts": {
                "configuration": "Closed-form difference between the rounded-throat and sharp-corner "
                                 "bell lengths for Rd/Rt = 0.382 over eps 5-50 and 60-100 percent: "
                                 "absolute dL/Rt (left, independent of eps) and relative dL/L_sharp "
                                 "(right); markers are the benchmark cases (open marker outside eps 5-50).",
                "evidence_class": "verification (closed form)",
                "not_established": "Performance consequence of either length."}},
        ],
        "wall_time_s": time.time() - t0,
        "notes": [
            "No raosim source, case file, tolerance or test was changed.",
            "Diagnostic reruns override length_pct in an in-memory copy of the case only.",
            "LR1 and LR2 answers (scout6_misc_refs, verified) incorporated 2026-11-21.",
        ],
    }
    for _f in result["figures"]:
        _f.update({"script": str(Path(__file__).resolve().relative_to(ROOT)), "command": result["command"],
                   "git_head": result["git_head"], "status": result["status"],
                   "wall_time_s": result["wall_time_s"], "hashes": "see result.hashes (script, raosim, inputs)",
                   "file_sha256": {q: sha256(ROOT / q) for q in _f["paths"]}})
    (OUT / "result.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"status": status, "repro": repro, "bench": py_bench["summary"],
                      "bench_failed": py_bench["failed"], "geom": py_geom["summary"],
                      "geom_failed": py_geom["failed"],
                      "diag": [(d["case_id"], d["statuses"]) for d in diag],
                      "inv": result["metrics"]["inventory_counts"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
