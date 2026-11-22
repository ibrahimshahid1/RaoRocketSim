"""E01: common-coordinate field error maps for the accepted t19 Nozzle-B refinement pair.

Opt-in experiment script (moc_extensions_plan_2026-11-21). Reads the banked NPZ arrays only.
Run: .venv-jax/bin/python -m tests.experiments.nozzle.moc_ext_e01_t19_error_maps
"""

from __future__ import annotations

import math

import numpy as np

from scripts import moc_ext_e00_common as C

# Pre-declared before the run.
ETA_BAND = 0.10            # near-wall band: wall distance / local wall radius
TOP_QUANTILE = 0.95        # "worst" nodes = |dtheta| >= p95
CONC_FRACTION_MIN = 0.80   # share of worst nodes inside the band (x >= x_D, eta <= ETA_BAND)
CONC_ENRICHMENT_MIN = 2.0  # and at least this multiple of the band's share of all nodes
RMS_OVER_MAX_MAX = 0.10    # "an order of magnitude below the maximum"
ETA_EDGES = np.array([0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0])
N_AXIAL_BINS = 12
REL_TOL = 1.0e-6
POSTHOC_DE_DIST = 0.05


def point_polyline_distance(pts, poly):
    a = poly[:-1][None, :, :]
    b = poly[1:][None, :, :]
    p = pts[:, None, :]
    ab = b - a
    t = np.clip(np.sum((p - a) * ab, axis=2) / np.maximum(np.sum(ab * ab, axis=2), 1e-300), 0.0, 1.0)
    proj = a + t[:, :, None] * ab
    return np.min(np.hypot(*(p - proj).transpose(2, 0, 1)), axis=1)


def wall_radius(x, wall):
    order = np.argsort(wall[:, 0])
    xs, idx = np.unique(wall[order, 0], return_index=True)
    return np.interp(x, xs, wall[order, 1][idx])


def summary(v):
    v = np.abs(np.asarray(v, dtype=float))
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "max": float(v.max()), "rms": float(math.sqrt(np.mean(v * v))),
            "p50": float(np.percentile(v, 50)), "p90": float(np.percentile(v, 90)),
            "p99": float(np.percentile(v, 99))}


def main():
    from scipy.interpolate import LinearNDInterpolator
    from lrekit.core.gas_dynamics import isentropic_pressure_ratio

    run = C.Run("E01", __file__)
    report = C.load_report()
    run.add_input(C.REPORT)
    for lvl in ("baseline", "refined"):
        run.add_input(C.NPZ[lvl], report["artifact_sha256"][f"{lvl}_field.npz"])
    data = {lvl: dict(np.load(C.NPZ[lvl], allow_pickle=False)) for lvl in ("baseline", "refined")}
    gamma = float(report["refined"]["config"]["gamma"])

    p_of = np.vectorize(lambda m: isentropic_pressure_ratio(float(m), gamma))
    fields = {}
    for lvl, d in data.items():
        fields[lvl] = {"xy": d["xy_over_rt"], "M": d["mach"], "theta": d["theta_rad"], "p": p_of(d["mach"])}
    interps = {lvl: {k: LinearNDInterpolator(f["xy"], f[k], fill_value=np.nan) for k in ("M", "theta", "p")}
               for lvl, f in fields.items()}

    per_target = {}
    cov = {}
    for target, source in (("baseline", "refined"), ("refined", "baseline")):
        f = fields[target]
        pred = {k: np.asarray(interps[source][k](f["xy"]), dtype=float) for k in ("M", "theta", "p")}
        finite = np.logical_and.reduce([np.isfinite(v) for v in pred.values()])
        cov[target] = float(np.mean(finite))
        p_alt = np.where(finite, p_of(np.where(finite, pred["M"], 2.0)), np.nan)
        wall = data[target]["wall_raw_m"]
        xy = f["xy"][finite]
        dist = point_polyline_distance(xy, wall)
        rw = wall_radius(xy[:, 0], wall)
        per_target[target] = {
            "xy": xy,
            "dM": pred["M"][finite] - f["M"][finite],
            "dtheta_deg": np.degrees(pred["theta"][finite] - f["theta"][finite]),
            "dp": pred["p"][finite] - f["p"][finite],
            "dp_alt": p_alt[finite] - f["p"][finite],
            "dist": dist,
            "eta": dist / rw,
            "x_D": float(data[target]["ce_x_m"][0]),
        }

    cat = {k: np.concatenate([per_target[t][k] for t in ("baseline", "refined")])
           for k in ("xy", "dM", "dtheta_deg", "dp", "dp_alt", "dist", "eta")}
    x_D = {t: per_target[t]["x_D"] for t in per_target}
    xD_node = np.concatenate([np.full(len(per_target[t]["dM"]), per_target[t]["x_D"]) for t in ("baseline", "refined")])

    ref = report["refinement"]["metrics"]
    measured = {
        "theta_max_abs_deg": float(np.max(np.abs(cat["dtheta_deg"]))),
        "mach_max_abs": float(np.max(np.abs(cat["dM"]))),
        "pressure_ratio_max_abs": float(np.max(np.abs(cat["dp"]))),
        "minimum_field_coverage": float(min(cov.values())),
        "comparison_coordinate_count": int(len(cat["dM"])),
        "mach_rms_abs": float(math.sqrt(np.mean(cat["dM"] ** 2))),
        "theta_rms_abs_deg": float(math.sqrt(np.mean(cat["dtheta_deg"] ** 2))),
        "pressure_ratio_rms_abs": float(math.sqrt(np.mean(cat["dp"] ** 2))),
        "theta_p99_abs_deg": float(np.percentile(np.abs(cat["dtheta_deg"]), 99)),
        "mach_p99_abs": float(np.percentile(np.abs(cat["dM"]), 99)),
    }
    repro = {}
    for k in ("theta_max_abs_deg", "mach_max_abs", "pressure_ratio_max_abs", "minimum_field_coverage"):
        rel_err = abs(measured[k] - ref[k]) / abs(ref[k])
        repro[k] = {"measured": measured[k], "report": ref[k], "rel_err": rel_err, "passes": rel_err <= REL_TOL}
    repro["comparison_coordinate_count"] = {"measured": measured["comparison_coordinate_count"],
                                            "report": ref["comparison_coordinate_count"],
                                            "passes": measured["comparison_coordinate_count"] == ref["comparison_coordinate_count"]}
    reproduced = all(v["passes"] for v in repro.values())

    alt_p = {"definition": "p/p0 evaluated from the interpolated Mach (instead of interpolating nodal p/p0)",
             "pressure_ratio_max_abs": float(np.max(np.abs(cat["dp_alt"]))),
             "note": "secondary; the acceptance uses nodal p/p0 interpolation as in the production comparison"}

    status = "PASS" if reproduced else "FAIL"
    result = {"status": status}
    if not reproduced:
        result.update({"reason": "report.json refinement metrics not reproduced to 1e-6 relative; E01 stopped",
                       "reproduction": repro})
        run.finish(result)
        print("E01 FAIL", repro)
        return 1

    ath = np.abs(cat["dtheta_deg"])
    aM = np.abs(cat["dM"])
    in_band = (cat["xy"][:, 0] >= xD_node) & (cat["eta"] <= ETA_BAND)
    thr = np.quantile(ath, TOP_QUANTILE)
    top = ath >= thr
    frac_top_in_band = float(np.mean(in_band[top]))
    band_share = float(np.mean(in_band))
    enrichment = frac_top_in_band / band_share if band_share > 0 else float("inf")
    sq_share_band = float(np.sum(ath[in_band] ** 2) / np.sum(ath ** 2))
    de_poly = np.concatenate([np.column_stack([data[t]["ce_x_m"], data[t]["ce_r_m"]]) for t in ("refined",)])
    d_de = point_polyline_distance(cat["xy"], de_poly)
    near_de = d_de <= POSTHOC_DE_DIST
    post_hoc = {"definition": f"post hoc, not pre-declared: nodes within {POSTHOC_DE_DIST} Rt of the 48/64 DE polyline",
                "fraction_top_near_DE": float(np.mean(near_de[top])), "near_DE_share_of_nodes": float(np.mean(near_de)),
                "fraction_top_downstream_of_D": float(np.mean(cat["xy"][top, 0] >= xD_node[top])),
                "downstream_of_D_share_of_nodes": float(np.mean(cat["xy"][:, 0] >= xD_node))}
    imax = int(np.argmax(ath))
    worst = {"x_over_rt": float(cat["xy"][imax, 0]), "r_over_rt": float(cat["xy"][imax, 1]),
             "eta": float(cat["eta"][imax]), "wall_distance_over_rt": float(cat["dist"][imax]),
             "dtheta_deg": float(cat["dtheta_deg"][imax])}
    rms_ratio = {"theta": measured["theta_rms_abs_deg"] / measured["theta_max_abs_deg"],
                 "mach": measured["mach_rms_abs"] / measured["mach_max_abs"]}
    conc_ok = frac_top_in_band >= CONC_FRACTION_MIN and enrichment >= CONC_ENRICHMENT_MIN
    rms_ok = all(v <= RMS_OVER_MAX_MAX for v in rms_ratio.values())

    eta_bins = []
    for lo, hi in zip(ETA_EDGES[:-1], ETA_EDGES[1:]):
        m = (cat["eta"] >= lo) & (cat["eta"] < hi)
        eta_bins.append({"eta_lo": float(lo), "eta_hi": float(hi), "dtheta_deg": summary(cat["dtheta_deg"][m]),
                         "dM": summary(cat["dM"][m]), "dp": summary(cat["dp"][m])})
    xe = np.linspace(cat["xy"][:, 0].min(), cat["xy"][:, 0].max() + 1e-12, N_AXIAL_BINS + 1)
    ax_bins = []
    for lo, hi in zip(xe[:-1], xe[1:]):
        m = (cat["xy"][:, 0] >= lo) & (cat["xy"][:, 0] < hi)
        ax_bins.append({"x_lo": float(lo), "x_hi": float(hi), "dtheta_deg": summary(cat["dtheta_deg"][m]),
                        "dM": summary(cat["dM"][m]), "dp": summary(cat["dp"][m])})

    figs = make_figure(run, cat, data, eta_bins, ax_bins)

    result.update({
        "evidence_class": "verification (numerical resolution of one constant-gamma configuration)",
        "configuration": {
            "inputs": [C.rel(C.NPZ["baseline"]), C.rel(C.NPZ["refined"]), C.rel(C.REPORT)],
            "levels": {"baseline": "n_control/n_kernel 32/48, kernel dtheta 0.0625 deg",
                       "refined": "n_control/n_kernel 48/64, kernel dtheta 0.03125 deg"},
            "gamma": gamma,
            "interpolation": "scipy.interpolate.LinearNDInterpolator on xy_over_rt, fill NaN; both directions",
            "pressure": "p/p0 = isentropic_pressure_ratio(M, gamma) at the nodes, then interpolated (production definition)",
            "coverage": "fraction of target nodes with finite interpolants (inside the other mesh's convex hull)",
            "eta_definition": "eta = Euclidean distance to the target mesh's wall_raw_m polyline / local wall radius r_w(x)",
            "band_definition": f"x >= x_D (target mesh DE start) and eta <= {ETA_BAND}",
            "predeclared": {"ETA_BAND": ETA_BAND, "TOP_QUANTILE": TOP_QUANTILE,
                            "CONC_FRACTION_MIN": CONC_FRACTION_MIN, "CONC_ENRICHMENT_MIN": CONC_ENRICHMENT_MIN,
                            "RMS_OVER_MAX_MAX": RMS_OVER_MAX_MAX, "REL_TOL": REL_TOL},
        },
        "metrics": {
            "reproduction": repro,
            "measured": measured,
            "coverage": cov,
            "x_D_over_rt": x_D,
            "pressure_variant": alt_p,
            "worst_theta_node": worst,
            "concentration": {"dtheta_p95_deg": float(thr), "n_top": int(np.sum(top)),
                              "fraction_top_in_band": frac_top_in_band, "band_share_of_nodes": band_share,
                              "enrichment": enrichment, "band_share_of_squared_dtheta": sq_share_band},
            "rms_over_max": rms_ratio,
            "post_hoc_DE_proximity": post_hoc,
            "eta_bins": eta_bins,
            "axial_bins": ax_bins,
        },
        "hypotheses": {
            "H_concentrated_near_wall_downstream_of_D": "supported" if conc_ok else "not supported",
            "H_rms_order_of_magnitude_below_max": "supported" if rms_ok else "not supported",
        },
        "figures": figs,
        "caption_facts": {
            "configuration": ("Accepted t19 Nozzle-B pair (gamma 1.23, epsilon 4.973, 64% length, Ru/Rt 1.5, "
                              "Rd/Rt 0.382, KL start line); bidirectional linear interpolation of the banked "
                              "32/48 and 48/64 BDE fields at 2546 common coordinates."),
            "evidence_class": "verification (numerical resolution of one constant-gamma configuration)",
            "not_established": "No continuum-error bound, no observed order, nothing about other geometries or gas models.",
        },
        "not_established": "No continuum-error bound, no observed order, nothing about other geometries or gas models.",
        "notes": [
            "Differences are interpolated-minus-target at each target node; both directions pooled as in report.json.",
            "Panel (c) bins by eta; the BDE region downstream of D spans only eta up to about 0.2, so the band test is "
            "reported with its enrichment over the band's node share.",
        ],
    })
    out = run.finish(result)
    print("E01", status, {k: v["rel_err"] for k, v in repro.items() if "rel_err" in v})
    print(" concentration", out["metrics"]["concentration"], "rms/max", rms_ratio)
    return 0


def make_figure(run, cat, data, eta_bins, ax_bins):
    plt = C.setup_style()
    import matplotlib.tri as mtri
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from matplotlib.gridspec import GridSpec

    cmap = LinearSegmentedColormap.from_list("ink_seq", ["#EEF4F4", "#8FBCC0", C.COLORS["teal"], C.COLORS["ink"]])
    xy = cat["xy"]
    key = np.round(xy, 12)
    _, uniq = np.unique(key, axis=0, return_index=True)
    uniq = np.sort(uniq)
    tri = mtri.Triangulation(xy[uniq, 0], xy[uniq, 1])
    t = tri.triangles
    pts = xy[uniq]
    edges = np.stack([np.hypot(*(pts[t[:, i]] - pts[t[:, (i + 1) % 3]]).T) for i in range(3)], axis=1)
    emax = edges.max(axis=1)
    tri.set_mask(emax > 4.0 * np.median(emax))

    wall = data["refined"]["wall_raw_m"]
    wall = wall[wall[:, 0] >= xy[:, 0].min() - 0.05]
    ce = (data["refined"]["ce_x_m"], data["refined"]["ce_r_m"])

    fig = plt.figure(figsize=(C.FULL_W, 8.2))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.0, 0.95], hspace=0.36, wspace=0.32,
                  left=0.09, right=0.93, top=0.975, bottom=0.06)
    specs = [("dtheta_deg", "|Δθ| (deg)", "a"), ("dM", "|ΔM| (–)", "b")]
    for row, (k, lab, letter) in enumerate(specs):
        ax = fig.add_subplot(gs[row, :])
        v = np.abs(cat[k][uniq])
        hi = float(v.max()) * 1.0001
        lo = max(hi * 1e-4, float(np.percentile(v[v > 0], 1)))
        levels = np.geomspace(lo, hi, 15)
        cs = ax.tricontourf(tri, np.clip(v, lo, hi), levels=levels, cmap=cmap, norm=LogNorm(lo, hi))
        ax.plot(wall[:, 0], wall[:, 1], color=C.COLORS["ink"], lw=1.0, label="wall (48/64)")
        ax.plot(ce[0], ce[1], color=C.COLORS["accent"], lw=1.0, ls="--", label="DE control surface (48/64)")
        imax = int(np.argmax(np.abs(cat[k])))
        ax.plot(xy[imax, 0], xy[imax, 1], marker="o", ms=4, mfc="none", mec=C.COLORS["accent"], mew=1.0,
                ls="none", label="maximum")
        ax.set_aspect("equal")
        ax.set_xlabel("x / R$_t$ (–)")
        ax.set_ylabel("r / R$_t$ (–)")
        cb = fig.colorbar(cs, ax=ax, pad=0.01, fraction=0.035)
        decades = [10.0 ** e for e in range(int(math.floor(math.log10(lo))), int(math.ceil(math.log10(hi))) + 1)]
        cb.set_ticks([d for d in decades if lo <= d <= hi])
        cb.set_ticklabels([f"$10^{{{int(round(math.log10(d)))}}}$" for d in decades if lo <= d <= hi])
        cb.ax.minorticks_off()
        cb.set_label(lab)
        cb.ax.tick_params(labelsize=7)
        if row == 0:
            ax.legend(loc="upper left", fontsize=7)
        C.panel_label(ax, letter)

    ax = fig.add_subplot(gs[2, 0])
    mids = np.array([0.5 * (b["eta_lo"] + b["eta_hi"]) for b in eta_bins])
    ok = np.array([b["dtheta_deg"]["n"] > 0 for b in eta_bins])
    for q, col, ls in (("p50", C.COLORS["muted"], ":"), ("p90", C.COLORS["teal"], "--"),
                       ("p99", C.COLORS["teal"], "-"), ("max", C.COLORS["ink"], "-")):
        vals = np.array([b["dtheta_deg"].get(q, np.nan) for b in eta_bins])
        ax.plot(mids[ok], vals[ok], color=col, ls=ls, marker="o", ms=3, label=q)
    ax.axvline(ETA_BAND, color=C.COLORS["accent"], lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("wall distance / local wall radius (–)")
    ax.set_ylabel("|Δθ| (deg)")
    ax.grid(True, which="major")
    ax.set_ylim(top=ax.get_ylim()[1] * 2.5)
    ax.legend(loc="upper left", ncol=4, fontsize=6.5, handlelength=1.6, columnspacing=0.8)
    C.panel_label(ax, "c")

    ax = fig.add_subplot(gs[2, 1])
    xm = np.array([0.5 * (b["x_lo"] + b["x_hi"]) for b in ax_bins])
    okx = np.array([b["dtheta_deg"]["n"] > 0 for b in ax_bins])
    for q, col, ls in (("rms", C.COLORS["muted"], ":"), ("p99", C.COLORS["teal"], "-"), ("max", C.COLORS["ink"], "-")):
        vals = np.array([b["dtheta_deg"].get(q, np.nan) for b in ax_bins])
        ax.plot(xm[okx], vals[okx], color=col, ls=ls, marker="o", ms=3, label=q)
    ax.axvline(float(data["refined"]["ce_x_m"][0]), color=C.COLORS["accent"], lw=0.8, ls="--")
    ax.text(float(data["refined"]["ce_x_m"][0]), 0.02, " x$_D$", transform=ax.get_xaxis_transform(),
            va="bottom", ha="left", fontsize=7, color=C.COLORS["accent"])
    ax.set_yscale("log")
    ax.set_xlabel("axial station x / R$_t$ (–)")
    ax.set_ylabel("|Δθ| (deg)")
    ax.grid(True, which="major")
    ax.set_ylim(top=ax.get_ylim()[1] * 2.5)
    ax.legend(loc="upper left", ncol=3, fontsize=6.5, handlelength=1.6, columnspacing=0.8)
    C.panel_label(ax, "d")
    return [C.save_figure(fig, run.out_dir, "e01_t19_error_maps")]


if __name__ == "__main__":
    raise SystemExit(main())
