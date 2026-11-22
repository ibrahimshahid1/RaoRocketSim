"""E07: forward bounded-net replay of the accepted Nozzle-B BVP walls (code-to-code diagnostic).

Opt-in experiment script (moc_extensions_plan_2026-11-21). ``replay_wall`` is the shared harness for E08/E09/E11.
Run one level:  .venv-jax/bin/python -m tests.experiments.nozzle.moc_ext_e07_forward_replay --level baseline
Assemble:       .venv-jax/bin/python -m tests.experiments.nozzle.moc_ext_e07_forward_replay --assemble
"""

from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

from scripts import moc_ext_e00_common as C

RT = 1.0
RD_OVER_RT = 0.382
RU_OVER_RT = 1.5
X_EXIT = 2.94
MASS_TOL = 0.02
LEVELS = {
    "baseline": {"n_kernel": 48, "dtheta_deg": 0.0625, "theta_B_deg": 27.758823633193966},
    "refined": {"n_kernel": 64, "dtheta_deg": 0.03125, "theta_B_deg": 27.760614752769474},
}
THRESH = {"mach_max_abs": 0.005, "theta_max_abs_deg": 0.05, "cf_rel": 1.0e-3}
N_MESH_RRC = 60
N_MESH_LRC = 40


def bell_nodes_from_solution(sol, arc):
    w = np.asarray(sol.wall_raw, dtype=float)
    keep = w[:, 0] >= arc.x_end - 1e-12 * max(arc.x_end, 1.0)
    xb, rb = w[keep, 0].copy(), w[keep, 1].copy()
    if abs(xb[0] - arc.x_end) > 1e-12:
        xb = np.concatenate([[arc.x_end], xb])
        rb = np.concatenate([[arc.r_end], rb])
    seam = {"first_bell_node": [float(xb[0]), float(rb[0])], "arc_end": [arc.x_end, arc.r_end],
            "dx": float(xb[0] - arc.x_end), "dr": float(rb[0] - arc.r_end)}
    xb[0], rb[0] = arc.x_end, arc.r_end
    return xb, rb, seam


def make_wall(theta_B, xb, rb, theta_E, Rd=RD_OVER_RT * RT):
    from lrekit.components.nozzle.solvers.bounded import CompositeWall
    from lrekit.components.nozzle.solvers.nasa import ArcWall
    from lrekit.components.nozzle.solvers.wall import SplineWall
    arc = ArcWall(RT, Rd, theta_B)
    bell = SplineWall(np.asarray(xb, float), np.asarray(rb, float), slope_start=math.tan(theta_B),
                      slope_end=math.tan(theta_E))
    return arc, bell, CompositeWall(arc, bell)


def exit_plane_cf(samples, gamma):
    from lrekit.core.gas_dynamics import isentropic_density_ratio, isentropic_pressure_ratio, isentropic_temperature_ratio
    s = sorted(samples, key=lambda d: d["r"])
    r = np.array([d["r"] for d in s])
    M = np.array([max(1.0, d["M"]) for d in s])
    th = np.array([d["theta"] for d in s])
    p = np.array([isentropic_pressure_ratio(m, gamma) for m in M])
    rho = np.array([isentropic_density_ratio(m, gamma) for m in M])
    T = np.array([isentropic_temperature_ratio(m, gamma) for m in M])
    flux = gamma * M * M * np.cos(th) ** 2 * p + p
    cf = float(2.0 * np.trapezoid(flux * r, r) / (RT * RT))
    dm = rho * M * np.sqrt(gamma * T) * np.cos(th) * r
    pm = float(np.trapezoid(p * dm, r) / np.trapezoid(dm, r))
    pa = float(np.trapezoid(p * r, r) / np.trapezoid(r, r))
    return cf, {"r": r, "M": M, "theta": th, "p": p, "mass_averaged_p_over_p0": pm, "area_averaged_p_over_p0": pa}


def replay_wall(theta_B, xb, rb, theta_E, *, n_kernel, dtheta_deg, gamma=C.GAMMA, Ru=RU_OVER_RT * RT,
                start_line=None, x_exit=X_EXIT, max_rrcs=5000, keep_net=False, starting_line_method="kliegel_levine"):
    """Forward bounded-net replay on throat arc + SplineWall(xb, rb). Returns a dict of compact results."""
    from lrekit.components.nozzle.solvers.bounded import crop_to_exit_plane, exit_plane_mass_flow, march_bounded_net
    from lrekit.components.nozzle.solvers.nasa import _make_throat_initial_line
    arc, bell, wall = make_wall(theta_B, xb, rb, theta_E)
    if start_line is None:
        start_line = _make_throat_initial_line(RT, Ru, theta_B, gamma, n_kernel, starting_line_method)
    t0 = time.time()
    net = march_bounded_net(start_line, wall, gamma, arc=arc, dtheta_limit=math.radians(dtheta_deg),
                            mass_tol=MASS_TOL, on_invalid="stop", x_exit=x_exit, max_rrcs=max_rrcs)
    t_march = time.time() - t0
    out = {
        "complete": bool(net.complete), "stop_reason": net.stop_reason, "n_rows": net.n_rows,
        "n_refined": net.n_refined, "containment_violations": net.containment_violations,
        "truncated_rows": net.truncated_rows, "fold_location": net.fold_location,
        "max_abs_mdot_rel_err": float(np.max(np.abs(net.mdot_rel_err))),
        "march_wall_time_s": t_march, "arc_x_end": arc.x_end, "arc_r_end": arc.r_end,
        "start_line": {"x": np.array([p.x for p in start_line]), "r": np.array([p.r for p in start_line]),
                       "M": np.array([p.M for p in start_line]), "theta": np.array([p.theta for p in start_line])},
        "wall": {k: np.array([getattr(row[0], k) for row in net.rows[1:]]) for k in ("x", "r", "M", "theta")},
        "mdot_start": float(net.massflow[0]),
    }
    last = net.rows[-1]
    out["last_row_end"] = {"x": float(last[-1].x), "r": float(last[-1].r)}
    if net.complete:
        samples = crop_to_exit_plane(net, x_exit)
        cf, ex = exit_plane_cf(samples, gamma)
        out["cf"] = cf
        out["exit_plane"] = ex
        out["mdot_exit"] = float(exit_plane_mass_flow(samples, gamma))
        out["mdot_exit_rel_to_start"] = (out["mdot_exit"] - out["mdot_start"]) / out["mdot_start"]
    if keep_net:
        out["net"] = net
    return out


def mesh_subsample(net, n_rrc=N_MESH_RRC, n_lrc=N_MESH_LRC):
    rows = net.rows
    js = np.unique(np.linspace(0, len(rows) - 1, n_rrc).round().astype(int))
    rrc = [np.array([[p.x, p.r] for p in rows[j]]) for j in js]
    child = []
    for j in range(1, len(rows)):
        m = {}
        for k, e in enumerate(net.parents[j]):
            if e is not None and e.get("cplus") is not None:
                m.setdefault(e["cplus"][1], k)
        child.append(m)
    starts = np.unique(np.linspace(1, len(rows[0]) - 1, n_lrc).round().astype(int))
    lrc = []
    for i0 in starts:
        pts = [[rows[0][i0].x, rows[0][i0].r]]
        i = i0
        for j in range(1, len(rows)):
            k = child[j - 1].get(i)
            if k is None:
                break
            pts.append([rows[j][k].x, rows[j][k].r])
            i = k
        lrc.append(np.array(pts))
    return rrc, lrc


def bvp_reference(sol, gamma):
    from lrekit.core.gas_dynamics import isentropic_pressure_ratio
    from lrekit.components.nozzle.solvers.nasa import full_control_surface_thrust
    from lrekit.components.nozzle.solvers.variational import _control_surface_flow_nodes
    bde = sol.construction_diagnostics["bde_artifacts"]
    wc = bde["bde_region"].wall_contour
    fct = full_control_surface_thrust(bde["kernel"], _control_surface_flow_nodes(sol.control_surface),
                                      gamma=gamma, Rt=RT, pa_over_p0=0.0)
    return {
        "wall": {"x": np.array([p.x for p in wc]), "r": np.array([p.r for p in wc]),
                 "M": np.array([p.M for p in wc]), "theta": np.array([p.theta for p in wc]),
                 "p": np.array([isentropic_pressure_ratio(p.M, gamma) for p in wc])},
        "cde": np.array([[p.x, p.r] for p in fct.CDE]),
        "cf": float(sol.thrust_coefficient),
        "B": [float(bde["kernel"].B.x), float(bde["kernel"].B.r)],
    }


def compare_wall(fwd_wall, ref_wall, x_lo):
    order = np.argsort(fwd_wall["x"])
    xf = fwd_wall["x"][order]
    xu, iu = np.unique(xf, return_index=True)
    sel = (ref_wall["x"] > x_lo) & (ref_wall["x"] <= xu[-1] + 1e-12)
    out = {"n_nodes": int(sel.sum()), "n_ref_nodes_beyond_forward_wall": int(np.sum(ref_wall["x"] > xu[-1] + 1e-12)),
           "forward_wall_x_max": float(xu[-1])}
    if not sel.any():
        out.update({"x": np.zeros(0), "dM": np.zeros(0), "dtheta_deg": np.zeros(0),
                    "mach_max_abs": None, "theta_max_abs_deg": None})
        return out
    xr = ref_wall["x"][sel]
    M_f = np.interp(xr, xu, fwd_wall["M"][order][iu])
    th_f = np.interp(xr, xu, fwd_wall["theta"][order][iu])
    dM = M_f - ref_wall["M"][sel]
    dth = np.degrees(th_f - ref_wall["theta"][sel])
    i = int(np.argmax(np.abs(dM)))
    k = int(np.argmax(np.abs(dth)))
    out.update({"x": xr, "dM": dM, "dtheta_deg": dth,
                "mach_max_abs": float(np.max(np.abs(dM))), "theta_max_abs_deg": float(np.max(np.abs(dth))),
                "worst_mach_x": float(xr[i]), "worst_theta_x": float(xr[k])})
    return out


def diagnose_fold(net, wall, gamma, dtheta_deg, x_exit=X_EXIT):
    """Rebuild the attempted next RRC without the fold audit and locate every reversed cell (read-only)."""
    from lrekit.components.nozzle.solvers.bounded import (_build_row, _calc_special_wall_point_at, _calc_wall_point_raw_general,
                                    _signed_cell_area, wall_x_at_theta)
    rows = net.rows
    orientation = 0.0
    for j in range(2, len(rows)):
        prev, new = rows[j - 1], rows[j]
        sflag = net.special[j]
        areas = []
        for i in range(1, min(len(new), len(prev))):
            ii_f = min(i if sflag else i + 1, len(prev) - 1)
            if ii_f >= 1:
                areas.append(_signed_cell_area(prev[ii_f - 1], prev[ii_f], new[i], new[i - 1]))
        nz = [a for a in areas if abs(a) > 0.0]
        if nz:
            orientation = math.copysign(1.0, float(np.median(nz)))
            break
    prev = rows[-1]
    dlim = math.radians(dtheta_deg)
    raw = _calc_wall_point_raw_general(prev, wall, gamma)
    if raw is None or isinstance(raw, str):
        return {"available": False, "reason": f"wall point {raw!r}"}
    dth = raw.theta - prev[0].theta
    if abs(dth) > dlim:
        th_t = prev[0].theta + math.copysign(0.5 * dlim, dth)
        x_t = wall_x_at_theta(wall, th_t, prev[0].x, raw.x)
        wpt, sflag = _calc_special_wall_point_at(prev, x_t, float(wall.r(x_t)), float(wall.theta(x_t)), gamma), True
    else:
        wpt, sflag = raw, False
    built = _build_row(prev, wpt, sflag, gamma, wall, wall_parent={"cplus": (len(rows) - 1, 1), "cminus": None},
                       row_index=len(rows), orientation=0.0)
    if built is None:
        return {"available": False, "reason": "row invalid without fold audit"}
    new = built[0]
    i_last_prev = len(prev) - 1
    rev = []
    for i in range(1, len(new) - 1):
        ii_f = min(i if sflag else i + 1, i_last_prev)
        if ii_f < 1:
            continue
        area = _signed_cell_area(prev[ii_f - 1], prev[ii_f], new[i], new[i - 1])
        if orientation * area <= 0.0:
            rev.append((i, float(new[i].x), float(new[i].r)))
    prev_x = np.array([p.x for p in prev])
    prev_r = np.array([p.r for p in prev])
    cross = np.where((prev_x[:-1] - x_exit) * (prev_x[1:] - x_exit) <= 0.0)[0]
    return {
        "rev_xy": np.array([[c[1], c[2]] for c in rev]).reshape(-1, 2),
        "attempted_row_xy": np.array([[p.x, p.r] for p in new]),
        "available": True, "orientation": orientation, "attempted_row_index": len(rows),
        "attempted_wall_point": {"x": float(wpt.x), "r": float(wpt.r), "theta_deg": math.degrees(wpt.theta),
                                 "M": float(wpt.M), "special": bool(sflag)},
        "previous_row_points": len(prev), "attempted_row_points": len(new),
        "previous_row_axis_end_x": float(prev_x[-1]),
        "previous_row_r_at_x_exit": [float(prev_r[c]) for c in cross],
        "n_reversed_cells": len(rev),
        "reversed_cells_min_x": min((c[1] for c in rev), default=None),
        "reversed_cells_max_x": max((c[1] for c in rev), default=None),
        "reversed_cells_all_downstream_of_exit": bool(rev) and all(c[1] > x_exit for c in rev),
        "first_reversed_cell": rev[0] if rev else None,
        "x_exit": x_exit,
    }


def kernel_identity(net, kernel):
    n = len(kernel.rrcs)
    rows_checked = min(n, net.n_rows)
    max_dM = max_dx = max_dr = 0.0
    len_mismatch = []
    for j in range(rows_checked):
        a, b = kernel.rrcs[j], net.rows[j]
        if len(a) != len(b):
            len_mismatch.append([j, len(a), len(b)])
            continue
        max_dM = max(max_dM, max(abs(p.M - q.M) for p, q in zip(a, b)))
        max_dx = max(max_dx, max(abs(p.x - q.x) for p, q in zip(a, b)))
        max_dr = max(max_dr, max(abs(p.r - q.r) for p, q in zip(a, b)))
    return {"kernel_rows": n, "rows_checked": rows_checked, "row_length_mismatches": len_mismatch[:20],
            "n_row_length_mismatches": len(len_mismatch), "max_abs_dM": max_dM, "max_abs_dx": max_dx,
            "max_abs_dr": max_dr,
            "bit_identical": bool(not len_mismatch and max_dM == 0.0 and max_dx == 0.0 and max_dr == 0.0),
            "net_row_after_kernel_wall_x": float(net.rows[n][0].x) if net.n_rows > n else None}


def run_level(level, max_rrcs):
    from lrekit.components.nozzle.solvers.nasa import ArcWall
    run = C.Run("E07", __file__)
    report = C.load_report()
    run.add_input(C.PKL[level], report["artifact_sha256"][f"{level}_solution.pkl"])
    sol = C.load_solution(level)
    lv = LEVELS[level]
    kernel = sol.construction_diagnostics["bde_artifacts"]["kernel"]
    theta_B = float(kernel.theta_B)
    assert abs(math.degrees(theta_B) - lv["theta_B_deg"]) < 1e-12
    assert abs(math.degrees(kernel.dtheta_limit_rad) - lv["dtheta_deg"]) < 1e-12
    arc = ArcWall(RT, RD_OVER_RT * RT, theta_B)
    xb, rb, seam = bell_nodes_from_solution(sol, arc)
    rep = replay_wall(theta_B, xb, rb, float(sol.theta_E), n_kernel=lv["n_kernel"], dtheta_deg=lv["dtheta_deg"],
                      max_rrcs=max_rrcs, keep_net=True)
    net = rep.pop("net")
    ident = kernel_identity(net, kernel)
    fold_diag = None
    if not net.complete:
        _, _, wall = make_wall(theta_B, xb, rb, float(sol.theta_E))
        fold_diag = diagnose_fold(net, wall, C.GAMMA, lv["dtheta_deg"])
    fold_arrays = {}
    if fold_diag and fold_diag.get("available"):
        fold_arrays = {"fold_rev_xy": fold_diag.pop("rev_xy"), "fold_attempted_row_xy": fold_diag.pop("attempted_row_xy")}
    kp = sol.kernel_points
    kp_row = net.rows[len(kernel.rrcs) - 1] if net.n_rows >= len(kernel.rrcs) else []
    kp_ident = (len(kp_row) == len(kp) and all(p.x == q.x and p.r == q.r and abs(p.M - max(q.M, 1.000001)) == 0.0
                                               for p, q in zip(kp, kp_row)))
    ref = bvp_reference(sol, C.GAMMA)
    cmp = compare_wall(rep["wall"], ref["wall"], arc.x_end)
    rrc, lrc = mesh_subsample(net)
    del net
    np.savez_compressed(run.out_dir / f"replay_{level}.npz",
                        fwd_wall_x=rep["wall"]["x"], fwd_wall_r=rep["wall"]["r"], fwd_wall_M=rep["wall"]["M"],
                        fwd_wall_theta=rep["wall"]["theta"],
                        ref_wall_x=ref["wall"]["x"], ref_wall_M=ref["wall"]["M"], ref_wall_theta=ref["wall"]["theta"],
                        ref_wall_r=ref["wall"]["r"], ref_cde=ref["cde"],
                        exit_r=rep.get("exit_plane", {}).get("r", np.zeros(0)),
                        exit_M=rep.get("exit_plane", {}).get("M", np.zeros(0)),
                        exit_theta=rep.get("exit_plane", {}).get("theta", np.zeros(0)),
                        cmp_x=cmp["x"], cmp_dM=cmp["dM"], cmp_dtheta_deg=cmp["dtheta_deg"],
                        bell_x=xb, bell_r=rb, **fold_arrays,
                        arc_wall_x=np.array([row[0].x for row in kernel.rrcs]),
                        arc_wall_M=np.array([row[0].M for row in kernel.rrcs]),
                        **{f"rrc_{i}": a for i, a in enumerate(rrc)}, **{f"lrc_{i}": a for i, a in enumerate(lrc)})
    cf_rel = (rep["cf"] - ref["cf"]) / ref["cf"] if "cf" in rep else None
    summary = {
        "level": level, "n_kernel": lv["n_kernel"], "dtheta_deg": lv["dtheta_deg"], "theta_B_deg": math.degrees(theta_B),
        "theta_E_deg": math.degrees(float(sol.theta_E)), "max_rrcs": max_rrcs, "bell_knots": int(len(xb)), "seam": seam,
        "complete": rep["complete"], "stop_reason": rep["stop_reason"], "n_rows": rep["n_rows"],
        "n_refined": rep["n_refined"], "containment_violations": rep["containment_violations"],
        "truncated_rows": rep["truncated_rows"], "fold_location": rep["fold_location"],
        "max_abs_mdot_rel_err": rep["max_abs_mdot_rel_err"], "march_wall_time_s": rep["march_wall_time_s"],
        "kernel_identity": ident, "kernel_points_identical": bool(kp_ident), "fold_diagnosis": fold_diag,
        "cf_forward": rep.get("cf"), "cf_bvp": ref["cf"], "cf_rel_diff": cf_rel,
        "mdot_start": rep["mdot_start"], "mdot_exit": rep.get("mdot_exit"),
        "mdot_exit_rel_to_start": rep.get("mdot_exit_rel_to_start"),
        "exit_plane_mass_averaged_p_over_p0": rep.get("exit_plane", {}).get("mass_averaged_p_over_p0"),
        "exit_plane_area_averaged_p_over_p0": rep.get("exit_plane", {}).get("area_averaged_p_over_p0"),
        "wall_compare": {k: v for k, v in cmp.items() if not isinstance(v, np.ndarray)},
        "wall_exit_forward": {"M": float(rep["wall"]["M"][-1]), "theta_deg": math.degrees(float(rep["wall"]["theta"][-1])),
                              "x": float(rep["wall"]["x"][-1])},
        "input_sha256": dict(run.inputs),
        "raosim_sources_sha256": C.raosim_sources(),
        "script_sha256": C.script_hashes(__file__),
        "git_head": C.git_head(),
    }
    with open(run.out_dir / f"level_{level}.json", "w") as f:
        json.dump(C.jsonable(summary), f, indent=1)
    print(json.dumps(C.jsonable({k: v for k, v in summary.items() if k not in ("seam",)}), indent=1)[:3000])


def assemble():
    import lrekit.components.nozzle.solvers.bounded, lrekit.components.nozzle.solvers.nasa, lrekit.components.nozzle.solvers.wall, lrekit.components.nozzle.solvers.variational  # noqa: F401
    run = C.Run("E07", __file__)
    report = C.load_report()
    levels = {}
    for lvl in ("baseline", "refined"):
        p = run.out_dir / f"level_{lvl}.json"
        if p.exists():
            levels[lvl] = json.load(open(p))
            run.add_input(C.PKL[lvl], report["artifact_sha256"][f"{lvl}_solution.pkl"])
            run.add_input(p)
            run.add_input(run.out_dir / f"replay_{lvl}.npz")
    complete = len(levels) == 2 and all(v["complete"] for v in levels.values())
    checks = {}
    for lvl, v in levels.items():
        if v["complete"] and v["wall_compare"]["mach_max_abs"] is not None:
            checks[lvl] = {"mach": v["wall_compare"]["mach_max_abs"] <= THRESH["mach_max_abs"],
                           "theta": v["wall_compare"]["theta_max_abs_deg"] <= THRESH["theta_max_abs_deg"],
                           "cf": abs(v["cf_rel_diff"]) <= THRESH["cf_rel"]}
    all_ok = complete and all(all(c.values()) for c in checks.values())
    status = "PASS-DIAGNOSTIC" if all_ok else "FAIL-DIAGNOSTIC"
    figs = make_figure(run, levels) if levels else []
    failure = {}
    for lvl, v in levels.items():
        if not v["complete"]:
            fd = v.get("fold_diagnosis") or {}
            failure[lvl] = {
                "stop_reason": v["stop_reason"], "rows_completed": v["n_rows"],
                "kernel_rows": v["kernel_identity"]["kernel_rows"], "fold_location_x_r": v["fold_location"],
                "attempted_wall_point": fd.get("attempted_wall_point"),
                "previous_row_axis_end_x": fd.get("previous_row_axis_end_x"),
                "n_reversed_cells": fd.get("n_reversed_cells"),
                "reversed_cells_x_range": [fd.get("reversed_cells_min_x"), fd.get("reversed_cells_max_x")],
                "reversed_cells_all_downstream_of_exit": fd.get("reversed_cells_all_downstream_of_exit"),
            }
    level_sources = {lvl: v.get("raosim_sources_sha256") for lvl, v in levels.items()}
    result = {
        "status": status,
        "level_raosim_sources_sha256": level_sources,
        "level_raosim_sources_consistent_with_assemble": {
            lvl: all(src[k] == C.raosim_sources()[k] for k in set(src) & set(C.raosim_sources()))
            for lvl, src in level_sources.items() if src},
        "level_march_wall_time_s": {lvl: v["march_wall_time_s"] for lvl, v in levels.items()},
        "failure_summary": failure,
        "evidence_class": "verification (code-to-code consistency of two repo algorithms on the same model and start line)",
        "configuration": {
            "wall": "nasa_moc.ArcWall(Rt=1, Rd=0.382, theta_B from the pickle kernel) + wall_model.SplineWall through "
                    "wall_raw nodes with x >= arc.x_end (first node snapped to the arc end), slope_start=tan(theta_B), "
                    "slope_end=tan(theta_E); moc_bounded.CompositeWall",
            "start_line": "nasa_moc._make_throat_initial_line(Rt=1, Ru=1.5, theta_B, 1.23, n_kernel, 'kliegel_levine')",
            "call": "march_bounded_net(start, wall, 1.23, arc=arc, dtheta_limit=radians(dtheta_deg), mass_tol=0.02, "
                    "on_invalid='stop', x_exit=2.94)",
            "cf": "exit-plane (x = 2.94) momentum plus pressure, pa = 0, from crop_to_exit_plane samples; trapezoid in r; "
                  "normalised by p0*pi*Rt^2",
            "wall_compare": "forward wall-point M and theta linearly interpolated in x to the BVP BDE wall nodes "
                            "(bde_region.wall_contour) with x > arc end",
            "levels": LEVELS, "diagnostic_thresholds": THRESH,
        },
        "metrics": {"levels": levels, "threshold_checks": checks},
        "hypotheses": {"H_forward_replay_reproduces_bvp": "supported" if all_ok else "not supported"},
        "figures": figs,
        "caption_facts": {
            "configuration": ("Forward bounded-net replay (lrekit.components.nozzle.solvers.bounded, on_invalid='stop') of the accepted t19 "
                              "Nozzle-B walls (gamma 1.23, 32/48 and 48/64) with the BVP's KL start line, throat arc and "
                              "turning limits. (a) wall Mach: forward net (stops at B) vs BVP BDE wall nodes; (b) 48/64 "
                              "net subsample, last kernel RRC to the axis, attempted next RRC and its reversed cells; "
                              "(c) axial distribution of reversed cells relative to the exit plane."),
            "evidence_class": "verification (code-to-code consistency of two repo algorithms on the same model and start line)",
            "not_established": "Independence from the shared KL start line and unit processes; physical validity.",
        },
        "not_established": "Independence from the shared KL start line and unit processes; physical validity.",
        "notes": ["Thresholds are the existing field-policy magnitudes used as pre-declared diagnostic thresholds, not gates.",
                  "Kernel rows (start line to B) are bit-identical to the pickled BVP kernel at both levels; kernel_points "
                  "(BD) identical.",
                  "Both levels stop with rrc_coalescence on the first post-kernel RRC. Every reversed cell lies downstream "
                  "of the exit plane (x > 2.94 Rt), on the part of the last kernel RRC that runs to the axis near x = 13 Rt "
                  "(the region the BVP already reports as a downstream characteristic caustic). The march audits whole "
                  "RRCs to the axis, so a fold outside the nozzle stops the replay; no wall/CF comparison is available.",
                  "The SplineWall interpolant between BVP wall nodes is part of the replay model; it is not the BVP's "
                  "own between-node wall representation."],
    }
    run.finish(result)
    print("E07", status, checks)


def make_figure(run, levels):
    plt = C.setup_style()
    from matplotlib.gridspec import GridSpec
    data = {lvl: dict(np.load(run.out_dir / f"replay_{lvl}.npz")) for lvl in levels}
    col = {"baseline": C.COLORS["teal"], "refined": C.COLORS["accent"]}
    lab = {"baseline": "32/48", "refined": "48/64"}
    fig = plt.figure(figsize=(C.FULL_W, 6.6))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.15, 0.9], hspace=0.6, wspace=0.32,
                  left=0.1, right=0.98, top=0.97, bottom=0.08)

    ax = fig.add_subplot(gs[0, :])
    for lvl, d in data.items():
        ax.plot(d["fwd_wall_x"], d["fwd_wall_M"], color=col[lvl], lw=1.1, label=f"forward net {lab[lvl]}")
        ax.plot(d["ref_wall_x"], d["ref_wall_M"], ls="none", marker="o", ms=2.5, mfc="none", mec=col[lvl],
                label=f"BVP BDE wall {lab[lvl]}")
    ax.axvline(float(data[next(iter(data))]["bell_x"][0]), color=C.COLORS["muted"], lw=0.6, ls=":")
    ax.text(float(data[next(iter(data))]["bell_x"][0]), 0.04, " B", transform=ax.get_xaxis_transform(),
            fontsize=7, color=C.COLORS["muted"])
    ax.set_xlabel("x / R$_t$ (–)")
    ax.set_ylabel("wall Mach number (–)")
    ax.set_xlim(0, 3.0)
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=6.5, ncol=2)
    C.panel_label(ax, "a")

    ax = fig.add_subplot(gs[1, :])
    lvl = "refined" if "refined" in data else next(iter(data))
    d = data[lvl]
    for k in sorted([k for k in d if k.startswith("rrc_")], key=lambda t: int(t[4:])):
        ax.plot(d[k][:, 0], d[k][:, 1], color="#AEB8BD", lw=0.3)
    for k in sorted([k for k in d if k.startswith("lrc_")], key=lambda t: int(t[4:])):
        ax.plot(d[k][:, 0], d[k][:, 1], color=C.COLORS["teal"], lw=0.3)
    last = d[sorted([k for k in d if k.startswith("rrc_")], key=lambda t: int(t[4:]))[-1]]
    ax.plot(last[:, 0], last[:, 1], color=C.COLORS["ink"], lw=0.9, label="last kernel RRC (BD, to the axis)")
    if "fold_attempted_row_xy" in d:
        ax.plot(d["fold_attempted_row_xy"][:, 0], d["fold_attempted_row_xy"][:, 1], color=C.COLORS["muted"], lw=0.7,
                ls="--", label="attempted next RRC")
        rv = d["fold_rev_xy"]
        ax.plot(rv[:, 0], rv[:, 1], ls="none", marker="x", ms=2.5, mew=0.6, color=C.COLORS["accent"],
                label="reversed cells (fold)")
    ax.plot(d["bell_x"], d["bell_r"], color=C.COLORS["ink"], lw=1.4, label="BVP wall")
    ax.plot(d["ref_cde"][:, 0], d["ref_cde"][:, 1], color=C.COLORS["accent"], lw=1.1, label="BVP C-D-E surface")
    ax.axvline(X_EXIT, color=C.COLORS["ink"], lw=0.7, ls=":")
    ax.text(X_EXIT, 2.3, " exit plane", fontsize=6.5, va="top", color=C.COLORS["ink"])
    ax.set_xlim(-0.1, 13.5)
    ax.set_ylim(0, 2.4)
    ax.set_xlabel("x / R$_t$ (–)")
    ax.set_ylabel("r / R$_t$ (–)")
    ax.legend(loc="upper right", fontsize=6.3, ncol=2)
    C.panel_label(ax, "b")

    ax = fig.add_subplot(gs[2, :])
    bins = np.linspace(0, 13.5, 55)
    for lvl, d in data.items():
        if "fold_rev_xy" in d and d["fold_rev_xy"].size:
            ax.hist(d["fold_rev_xy"][:, 0], bins=bins, histtype="step", color=col[lvl], lw=1.0,
                    label=f"{lab[lvl]}: {len(d['fold_rev_xy'])} reversed cells")
    ax.axvline(X_EXIT, color=C.COLORS["ink"], lw=0.7, ls=":")
    ax.set_xlabel("axial position of reversed cell x / R$_t$ (–)")
    ax.set_ylabel("cells per bin (–)")
    ax.set_xlim(-0.1, 13.5)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right", fontsize=6.5)
    C.panel_label(ax, "c")
    return [C.save_figure(fig, run.out_dir, "e07_forward_replay")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=list(LEVELS))
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--max-rrcs", type=int, default=5000)
    a = ap.parse_args()
    if a.level:
        run_level(a.level, a.max_rrcs)
    if a.assemble:
        assemble()


if __name__ == "__main__":
    main()
