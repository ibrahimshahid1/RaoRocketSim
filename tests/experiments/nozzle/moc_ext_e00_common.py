"""Shared provenance, input and figure helpers for the opt-in moc_ext_e## experiment scripts.

Not imported by any library or default path.
"""

from __future__ import annotations

from lrekit.paths import package_root, repository_root as _repository_root

import hashlib
import json
import math
import pickle
import subprocess
import sys
import time
from pathlib import Path

REPO = _repository_root()
OUT_ROOT = REPO / "builds/moc_extensions_plan_2026-11-21/opus_b"
T19_DOCS = REPO / "docs/subsystems/S06-rao-moc-bvp-contour/audit_2026-11-19/evidence/validation/t19_refinement"
T19_BUILD = REPO / "builds/moc_audit_2026-11-19/sol_b/t19_refinement"
REPORT = T19_DOCS / "report.json"
NPZ = {"baseline": T19_DOCS / "baseline_field.npz", "refined": T19_DOCS / "refined_field.npz"}
PKL = {"baseline": T19_BUILD / "baseline_solution.pkl", "refined": T19_BUILD / "refined_solution.pkl"}
GAMMA = 1.23
COLORS = {"ink": "#19333C", "teal": "#176D76", "muted": "#51616B", "accent": "#C26A2E"}
FULL_W = 6.5
HALF_W = 3.2

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path) -> str:
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def git_modified(paths) -> list[str]:
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout
    except Exception:
        return []
    return [line[3:] for line in out.splitlines() if line.strip()]


def load_report() -> dict:
    with open(REPORT) as f:
        return json.load(f)


def load_solution(level: str):
    expected = load_report()["artifact_sha256"][f"{level}_solution.pkl"]
    actual = sha256(PKL[level])
    if actual != expected:
        raise RuntimeError(f"{level} pickle sha256 {actual} != report {expected}")
    with open(PKL[level], "rb") as f:
        return pickle.load(f)


def verify_npz(level: str) -> str:
    expected = load_report()["artifact_sha256"][f"{level}_field.npz"]
    actual = sha256(NPZ[level])
    if actual != expected:
        raise RuntimeError(f"{level} npz sha256 {actual} != report {expected}")
    return actual


def raosim_sources() -> dict:
    out = {}
    root = package_root().resolve()
    for mod in list(sys.modules.values()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        p = Path(f).resolve()
        if p.suffix == ".py" and root in p.parents:
            out[rel(p)] = sha256(p)
    return dict(sorted(out.items()))


def script_hashes(*paths) -> dict:
    files = {Path(__file__).resolve(), *[Path(p).resolve() for p in paths]}
    return {rel(p): sha256(p) for p in sorted(files)}


def setup_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.edgecolor": COLORS["muted"],
        "axes.labelcolor": COLORS["ink"],
        "xtick.color": COLORS["muted"],
        "ytick.color": COLORS["muted"],
        "text.color": COLORS["ink"],
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "grid.color": "#D5DBDE",
        "grid.linewidth": 0.4,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "savefig.dpi": 200,
    })
    return plt


def panel_label(ax, letter):
    ax.text(0.0, 1.02, f"({letter})", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8.5, fontweight="bold", color=COLORS["ink"])


def save_figure(fig, out_dir: Path, stem: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    return {"pdf": rel(pdf), "png": rel(png), "pdf_sha256": sha256(pdf), "png_sha256": sha256(png)}


def jsonable(obj):
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None if math.isnan(obj) else ("inf" if obj > 0 else "-inf")
    return obj


class Run:
    def __init__(self, exp_id: str, script: str):
        self.exp_id = exp_id
        self.script = Path(script).resolve()
        self.t0 = time.time()
        self.out_dir = OUT_ROOT / exp_id.lower()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.inputs: dict = {}

    def add_input(self, path, expected: str | None = None):
        h = sha256(path)
        if expected is not None and h != expected:
            raise RuntimeError(f"input {path} sha256 {h} != expected {expected}")
        self.inputs[rel(path)] = h
        return h

    def finish(self, result: dict, extra_scripts=()) -> dict:
        sources = raosim_sources()
        base = {
            "id": self.exp_id,
            "generated_by": rel(self.script),
            "command": " ".join([rel(sys.executable)] + sys.orig_argv[1:])
            if hasattr(sys, "orig_argv") else " ".join(sys.argv),
            "git_head": git_head(),
            "script_sha256": script_hashes(self.script, *extra_scripts),
            "raosim_sources_sha256": sources,
            "raosim_sources_modified_vs_head": git_modified(list(sources)) if sources else [],
            "input_sha256": self.inputs,
            "wall_time_s": round(time.time() - self.t0, 3),
        }
        base.update(result)
        out = jsonable(base)
        with open(self.out_dir / "result.json", "w") as f:
            json.dump(out, f, indent=1)
        return out
