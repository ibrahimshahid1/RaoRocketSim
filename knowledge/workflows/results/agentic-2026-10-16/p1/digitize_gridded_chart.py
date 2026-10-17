"""Dense, continuity-checked digitization of five ordered curves per panel.

Reviewer measurement from an original-page render; NOT a source-stated value.
"""
import json, sys
import numpy as np
from PIL import Image

def runs(mask):
    idx = np.where(mask)[0]
    if not len(idx):
        return []
    out, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            out.append((start, prev)); start = prev = i
    out.append((start, prev))
    return out

def lin(v, xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float); o = np.argsort(xs)
    return float(np.interp(v, xs[o], ys[o]))

cfg = json.load(open(sys.argv[1]))
im = np.array(Image.open(cfg["image"]).convert("L")).astype(int)
dark = im < cfg.get("threshold", 110)
gx, gv = cfg["x_grid_px"], cfg["x_grid_val"]
ncurves = cfg.get("ncurves", 5)
wmin, wmax = cfg.get("run_width", [4, 15])
result = {"image": cfg["image"], "method": "dense column scan; gridline rows/cols masked +-4 px; columns with exactly %d runs of width %d-%d px kept; median within +-%d px" % (ncurves, wmin, wmax, cfg.get("window", 12)), "panels": {}}
for name, p in cfg["panels"].items():
    cols = {}
    x_lo, x_hi = int(gx[1]) + 3, int(gx[-1]) - 3
    for x in range(x_lo, x_hi):
        if min(abs(x - g) for g in gx) <= 4:
            continue
        col = dark[p["y_top"]:p["y_bot"], x].copy()
        if cfg.get("mask_grid_rows", False):
            for g in p["y_grid_px"]:
                lo, hi = int(g - 4 - p["y_top"]), int(g + 4 - p["y_top"])
                col[max(lo, 0):max(hi, 0)] = False
        merged = runs(col)
        # Curves are thicker strokes than gridlines; a curve lying on a gridline stays one run.
        good = [((a + b) / 2 + p["y_top"], b - a + 1) for a, b in merged if wmin <= b - a + 1 <= wmax]
        if len(good) == ncurves:
            cols[x] = [lin(y, p["y_grid_px"], p["y_grid_val"]) for y, _ in good]
    res = {}
    for eps in cfg["eps_targets"]:
        x = lin(eps, gv, gx)
        near = [v for c, v in cols.items() if abs(c - x) <= cfg.get("window", 12)]
        if eps == gv[1]:
            near = [v for c, v in cols.items() if 0 < c - x <= 12]
        if not near:
            res[str(eps)] = {"state": "NO_CLEAN_COLUMNS"}
            continue
        arr = np.array(near)
        res[str(eps)] = {"columns": len(near), "median": [round(float(m), 2) for m in np.median(arr, axis=0)],
                         "spread": [round(float(s), 2) for s in (arr.max(axis=0) - arr.min(axis=0))]}
    result["panels"][name] = {"clean_columns": len(cols), "readings": res}
print(json.dumps(result, indent=1))
