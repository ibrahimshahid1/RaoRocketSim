"""Digitize SP-8120 Fig. 5(b): 10 ordered curves (5 alpha_i above 5 alpha_e) on a log-epsilon axis.
Reviewer measurement from an original-page render; NOT a source-stated value."""
import json, math, sys
import numpy as np
from PIL import Image
img = sys.argv[1]
im = np.array(Image.open(img).convert("L")).astype(int)
dark = im < 110
tick_x = [264.5, 353.5, 415.5, 465.0, 615.5, 766.0, 855.5, 919.5]
tick_lnv = [math.log(v) for v in (4, 6, 8, 10, 20, 40, 60, 80)]
y_px = [887.0, 682.0, 478.5, 274.5, 69.0]; y_val = [0, 10, 20, 30, 40]
def x_of(eps):
    return float(np.interp(math.log(eps), tick_lnv, tick_x, left=None))
def eps_of(x):
    return math.exp(float(np.interp(x, tick_x, tick_lnv)))
def deg(y):
    return float(np.interp(y, y_px[::-1], y_val[::-1]))
def runs(mask):
    idx = np.where(mask)[0]
    if not len(idx): return []
    out, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i == p + 1: p = i
        else: out.append((s, p)); s = p = i
    out.append((s, p)); return out
cols = {}
for x in range(300, 975):
    if min(abs(x - t) for t in tick_x) <= 3:
        pass  # ticks only reach ~35 px above the axis, below every curve; keep the column
    col = dark[60:860, x]
    rs = [((a + b) / 2 + 60, b - a + 1) for a, b in runs(col) if 4 <= b - a + 1 <= 16]
    if len(rs) == 10:
        cols[x] = [deg(y) for y, _ in rs]
targets = [6, 8, 10, 14, 15, 20, 25, 30, 35, 40, 45, 50]
out = {"image": img, "clean_columns": len(cols), "x_range_eps": [round(eps_of(min(cols)), 2), round(eps_of(max(cols)), 2)] if cols else None,
       "method": "log-x calibrated from ticks 4..80; columns with exactly 10 runs (width 4-16 px); median within +-10 px", "readings": {}}
for e in targets:
    x = x_of(e)
    near = [v for c, v in cols.items() if abs(c - x) <= 10]
    if not near:
        out["readings"][str(e)] = {"state": "NO_CLEAN_COLUMNS"}; continue
    arr = np.array(near)
    med = np.median(arr, axis=0); spr = arr.max(axis=0) - arr.min(axis=0)
    out["readings"][str(e)] = {"n": len(near), "alpha_i": [round(float(v), 2) for v in med[:5]],
                               "alpha_e": [round(float(v), 2) for v in med[5:]],
                               "spread": [round(float(v), 2) for v in spr]}
print(json.dumps(out, indent=1))

# --- smooth per-curve fit in ln(eps) with iterative outlier rejection ---
xs = np.array(sorted(cols)); Y = np.array([cols[x] for x in xs]); L = np.log([eps_of(x) for x in xs])
fits = {}
for k in range(10):
    keep = np.ones(len(xs), bool)
    for _ in range(4):
        coef = np.polyfit(L[keep], Y[keep, k], 3)
        resid = Y[:, k] - np.polyval(coef, L)
        new = np.abs(resid) < max(0.5, 3 * np.std(resid[keep]))
        if (new == keep).all(): break
        keep = new
    fits[k] = {"coef": coef.tolist(), "rms": float(np.sqrt(np.mean(resid[keep] ** 2))), "kept": int(keep.sum()),
               "rejected": int((~keep).sum())}
smooth = {}
for e in [5, 6, 8, 10, 14, 15, 20, 25, 30, 35, 40, 45, 50]:
    vals = [round(float(np.polyval(fits[k]["coef"], math.log(e))), 2) for k in range(10)]
    smooth[str(e)] = {"alpha_i": vals[:5], "alpha_e": vals[5:]}
print(json.dumps({"fit_quality": {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in f.items() if kk != "coef"} for k, f in fits.items()}, "smoothed": smooth}, indent=1))
