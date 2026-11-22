"""E07b (DIAGNOSTIC): exit-plane-limited forward replay of the accepted Nozzle-B walls.

Harness variant (script-local, not production): march_bounded_net with (i) each RRC truncated after its first point
with x > x_exit (kept for interpolation, no axis closure), (ii) the fold/invalid-cell audit applied only to cells whose
four corners have x <= x_exit (reversed cells beyond are counted, not fatal), (iii) rows reaching the axis at
x <= x_exit unchanged. E07 keeps its FAIL-DIAGNOSTIC record.
Run: .venv-jax/bin/python -m tests.experiments.nozzle.moc_ext_e07b_exit_limited_replay
"""

from __future__ import annotations

import json
import math
import time

import numpy as np

from scripts import moc_ext_e00_common as C

X_EXIT = 2.94
MASS_TOL = 0.02
DOD_BAND = 0.2
E05_CASE = {"alpha_deg": 15.0, "gamma": 1.4, "n": 49}
CHAR_FIELDS = ("x", "r", "theta", "M", "nu", "mu", "compat_plus", "compat_minus")


def build_row_limited(prev, wall_pt, sflag, gamma, wall, *, wall_parent, row_index, orientation, x_exit, stats):
    from lrekit.components.nozzle.solvers.bounded import _CellFold, _signed_cell_area
    from lrekit.components.nozzle.solvers.nasa import _calc_axial_mesh_point, _calc_interior_mesh_point
    new = [wall_pt]
    parents = [wall_parent]
    i_last_prev = len(prev) - 1
    i_end = i_last_prev + (1 if sflag else 0)
    prev_exit_truncated = prev[-1].x > x_exit and abs(prev[-1].r) > 1e-12
    truncated = False
    violations = 0
    if wall_pt.x > x_exit:
        return new, parents, False, 0, True
    for i in range(1, i_end):
        interior, negative_r = _calc_interior_mesh_point(prev, new, i, sflag, gamma)
        if negative_r:
            truncated = True
            break
        if interior is None:
            return None
        ii_f = min(i if sflag else i + 1, i_last_prev)
        p2 = new[i - 1]
        inside = interior.x <= x_exit and p2.x <= x_exit and (
            ii_f < 1 or (prev[ii_f - 1].x <= x_exit and prev[ii_f].x <= x_exit))
        area_rev = False
        if orientation != 0.0 and ii_f >= 1:
            area = _signed_cell_area(prev[ii_f - 1], prev[ii_f], interior, new[i - 1])
            edge = math.hypot(prev[ii_f].x - prev[ii_f - 1].x, prev[ii_f].r - prev[ii_f - 1].r) * math.hypot(
                interior.x - prev[ii_f].x, interior.r - prev[ii_f].r)
            area_rev = orientation * area <= 1e-12 * max(edge, 1e-24)
        upstream = interior.x < p2.x - 1e-12 * max(abs(p2.x), 1.0)
        if inside:
            if interior.M < 1.0001:
                return None
            if area_rev:
                raise _CellFold(i, interior.x, interior.r)
            if upstream:
                return None
            if interior.x <= wall.x_end and (
                    interior.r > wall.r(interior.x) + 1e-9 * max(abs(interior.r), 1.0)):
                violations += 1
        else:
            stats["beyond_subsonic"] += int(interior.M < 1.0001)
            stats["beyond_reversed"] += int(area_rev)
            stats["beyond_upstream"] += int(upstream)
            if area_rev and stats["first_beyond_reversed"] is None:
                stats["first_beyond_reversed"] = (row_index, float(interior.x), float(interior.r))
        ii = min(i if sflag else i + 1, i_last_prev)
        new.append(interior)
        parents.append({"cplus": (row_index - 1, ii), "cminus": (row_index, i - 1)})
        if interior.x > x_exit:
            return new, parents, False, violations, True
    if prev_exit_truncated and not truncated:
        stats["dod_violation_rows"].append(row_index)
        return "dod"
    axis_pt = _calc_axial_mesh_point(new, gamma)
    if axis_pt is None:
        if truncated:
            return new, parents, True, violations, False
        return None
    new.append(axis_pt)
    parents.append({"cplus": None, "cminus": (row_index, len(new) - 2)})
    return new, parents, truncated, violations, False


def march_limited(start_axis_first, wall, gamma, *, arc=None, dtheta_limit=math.radians(0.5), mass_tol=0.02,
                  on_invalid="stop", x_exit=None, max_rrcs=5000):
    """Copy of lrekit.components.nozzle.solvers.bounded.march_bounded_net with the exit-plane limits (i)-(iii)."""
    from lrekit.components.nozzle.solvers.bounded import (BoundedNet, CharacteristicCellError, _calc_special_wall_point_at,
                                    _calc_wall_point_raw_general, _CellFold, _signed_cell_area,
                                    _wall_first_nodes, wall_x_at_theta)
    from lrekit.components.nozzle.solvers.nasa import _calc_arc_wall_point_with_special, calc_massflow_along_rrc
    x_exit = float(wall.x_end if x_exit is None else x_exit)
    stats = {"beyond_subsonic": 0, "beyond_reversed": 0, "beyond_upstream": 0, "first_beyond_reversed": None,
             "dod_violation_rows": [], "exit_truncated_rows": 0, "mass_checked_rows": 0, "mass_unchecked_rows": 0}
    rows = [list(reversed(start_axis_first))]
    parents = [None]
    special = [False]
    massflow = [float(calc_massflow_along_rrc(_wall_first_nodes(rows[0], gamma), gamma)[0])]
    mdot_rel_err = [0.0]
    mdot0 = massflow[0]
    n_refined = violations = truncated_rows = 0
    complete = False
    stop_reason = "max_rrcs"
    exit_trunc_flags = [False]

    def finish(reason, ok=False, fold=None):
        net = BoundedNet(rows=rows, parents=parents, special=special, massflow=massflow, mdot_rel_err=mdot_rel_err,
                         gamma=gamma, x_exit=x_exit, complete=ok, stop_reason=reason, n_refined=n_refined,
                         containment_violations=violations, truncated_rows=truncated_rows,
                         wall_x=[float(r[0].x) for r in rows[1:]])
        net.fold_location = fold
        net.limited_stats = stats
        net.exit_truncated = exit_trunc_flags
        return net

    seam = float(arc.x_end) if arc is not None else -math.inf
    orientation = 0.0
    for j in range(1, max_rrcs + 1):
        prev = rows[-1]
        if len(prev) < (2 if exit_trunc_flags[-1] else 3):
            return finish("row_too_short")
        prev_wall = prev[0]
        on_arc = arc is not None and (prev_wall.x < seam - 1e-12 * max(seam, 1.0)
                                      and prev_wall.theta < arc.theta_max - 1e-9)
        attempts = []
        if on_arc:
            res = _calc_arc_wall_point_with_special(prev, arc, gamma, dtheta_limit)
            if res is None:
                return finish("arc_wall_point_failed")
            wall_pt, sflag = res
            attempts.append((wall_pt, sflag, "special" if sflag else "raw"))
        else:
            raw = _calc_wall_point_raw_general(prev, wall, gamma)
            if raw is None:
                return finish("wall_point_failed")
            if isinstance(raw, str):
                x3 = x_exit
                sp = _calc_special_wall_point_at(prev, x3, float(wall.r(x3)), float(wall.theta(x3)), gamma)
                if sp is None:
                    return finish("exit_lip_point_failed")
                attempts.append((sp, True, "exit_lip"))
            else:
                dth = raw.theta - prev_wall.theta
                if raw.x <= prev_wall.x + 1e-12 * max(abs(prev_wall.x), 1.0):
                    if on_invalid == "reject":
                        raise CharacteristicCellError(f"wall point of RRC {j} lands upstream of RRC {j-1}")
                    return finish("wall_point_upstream")
                if abs(dth) > dtheta_limit:
                    th_t = prev_wall.theta + math.copysign(0.5 * dtheta_limit, dth)
                    x_t = wall_x_at_theta(wall, th_t, prev_wall.x, raw.x)
                    sp = _calc_special_wall_point_at(prev, x_t, float(wall.r(x_t)), float(wall.theta(x_t)), gamma)
                    if sp is None:
                        return finish("special_wall_point_failed")
                    attempts.append((sp, True, "special"))
                else:
                    attempts.append((raw, False, "raw"))
            if on_invalid == "refine" and attempts[0][2] == "raw":
                raw_pt = attempts[0][0]
                th_t = prev_wall.theta + 0.5 * (raw_pt.theta - prev_wall.theta)
                x_t = wall_x_at_theta(wall, th_t, prev_wall.x, raw_pt.x)
                if x_t > prev_wall.x:
                    sp = _calc_special_wall_point_at(prev, x_t, float(wall.r(x_t)), float(wall.theta(x_t)), gamma)
                    if sp is not None:
                        attempts.append((sp, True, "refined"))
        built = None
        used_sflag = False
        fold = None
        for k, (wall_pt, sflag, foot) in enumerate(attempts):
            if wall_pt.r < 0.0 or wall_pt.M < 1.0001:
                continue
            try:
                built = build_row_limited(prev, wall_pt, sflag, gamma, wall,
                                          wall_parent={"cplus": (j - 1, 1), "cminus": None, "foot": foot},
                                          row_index=j, orientation=orientation if j >= 3 else 0.0,
                                          x_exit=x_exit, stats=stats)
            except _CellFold as exc:
                fold = exc
                built = None
            if built == "dod":
                return finish("domain_of_dependence_violation")
            if built is not None:
                used_sflag = bool(sflag)
                if k > 0:
                    n_refined += 1
                break
        if built is None:
            if fold is not None:
                if on_invalid == "reject":
                    raise CharacteristicCellError(f"RRC {j} crosses RRC {j-1} at x={fold.x:.6g}, r={fold.r:.6g}")
                return finish("rrc_coalescence", fold=(float(fold.x), float(fold.r)))
            if on_invalid == "reject":
                raise CharacteristicCellError(f"RRC {j} has an invalid cell")
            return finish("invalid_cell")
        new_row, new_parents, truncated, viol, exit_trunc = built
        violations += viol
        if orientation == 0.0 and j >= 2:
            areas = []
            for i in range(1, min(len(new_row), len(prev))):
                ii_f = min(i if used_sflag else i + 1, len(prev) - 1)
                if ii_f >= 1:
                    areas.append(_signed_cell_area(prev[ii_f - 1], prev[ii_f], new_row[i], new_row[i - 1]))
            nz = [a for a in areas if abs(a) > 0.0]
            if nz:
                orientation = math.copysign(1.0, float(np.median(nz)))
        if truncated:
            truncated_rows += 1
        if exit_trunc:
            stats["exit_truncated_rows"] += 1
            stats["mass_unchecked_rows"] += 1
            mdot, rel = float("nan"), float("nan")
        else:
            stats["mass_checked_rows"] += 1
            mdot = float(calc_massflow_along_rrc(_wall_first_nodes(new_row, gamma), gamma)[0])
            rel = (mdot - mdot0) / mdot0 if mdot0 > 0 else 0.0
            if abs(rel) > mass_tol:
                return finish("massflow_drift")
        rows.append(new_row)
        parents.append(new_parents)
        special.append(used_sflag)
        massflow.append(mdot)
        mdot_rel_err.append(rel)
        exit_trunc_flags.append(bool(exit_trunc))
        if new_row[0].x >= x_exit - 1e-12 * max(abs(x_exit), 1.0):
            complete = True
            stop_reason = "exit_reached"
            break
    return finish(stop_reason, ok=complete)


def dod_margins(net, x_exit, band=DOD_BAND, extra_nodes=()):
    """theta + mu < 90 deg and theta - mu > -90 deg margins (deg) for retained nodes near x_exit and first-beyond nodes."""
    out = {}
    groups = {"retained_within_band": [], "first_beyond_kept": [], "next_dropped_from_full_march": list(extra_nodes)}
    for row in net.rows[1:]:
        for p in row:
            if x_exit - band <= p.x <= x_exit:
                groups["retained_within_band"].append(p)
        if row[-1].x > x_exit and abs(row[-1].r) > 1e-12:
            groups["first_beyond_kept"].append(row[-1])
    for name, pts in groups.items():
        if not pts:
            out[name] = {"n": 0}
            continue
        th = np.array([p.theta for p in pts])
        mu = np.array([p.mu for p in pts])
        mp = 90.0 - np.degrees(th + mu)
        mm = np.degrees(th - mu) + 90.0
        ip, im = int(np.argmin(mp)), int(np.argmin(mm))
        out[name] = {"n": len(pts), "min_margin_theta_plus_mu_deg": float(mp[ip]),
                     "at_plus": [float(pts[ip].x), float(pts[ip].r)],
                     "min_margin_theta_minus_mu_deg": float(mm[im]), "at_minus": [float(pts[im].x), float(pts[im].r)],
                     "all_satisfied": bool(np.all(mp > 0) and np.all(mm > 0))}
    return out


def harness_verification():
    from lrekit.components.nozzle.solvers.bounded import march_bounded_net
    from tests.experiments.nozzle.moc_ext_e05_source_flow_oracle import build_case
    c = E05_CASE
    start, wall, meta = build_case(c["alpha_deg"], c["gamma"], c["n"])
    g = c["gamma"]
    x_mid = 0.5 * (wall.x_start + wall.x_end)
    kw = dict(arc=None, dtheta_limit=math.radians(0.5), mass_tol=0.02, on_invalid="stop")
    full = march_bounded_net(start, wall, g, **kw)
    mid = march_bounded_net(start, wall, g, x_exit=x_mid, **kw)
    lim = march_limited(start, wall, g, x_exit=x_mid, **kw)

    def compare(ref, all_kept=False):
        n_cmp = 0
        mism = []
        missing = 0
        for j in range(lim.n_rows):
            for k, p in enumerate(lim.rows[j]):
                if p.x > x_mid and not all_kept:
                    continue
                if j >= ref.n_rows or k >= len(ref.rows[j]):
                    missing += 1
                    continue
                q = ref.rows[j][k]
                n_cmp += 1
                if any(getattr(p, f) != getattr(q, f) for f in CHAR_FIELDS):
                    mism.append([j, k])
        return {"nodes_compared": n_cmp, "mismatches": len(mism), "first_mismatches": mism[:10],
                "nodes_missing_in_reference": missing, "bit_identical": n_cmp > 0 and not mism and missing == 0}

    nxt = []
    for j in range(1, lim.n_rows):
        row = lim.rows[j]
        if row[-1].x > x_mid and abs(row[-1].r) > 1e-12 and j < full.n_rows and len(full.rows[j]) > len(row):
            nxt.append(full.rows[j][len(row)])
    res = {
        "case": dict(c, x_start=wall.x_start, x_end=wall.x_end, x_exit_mid=x_mid),
        "full_march": {"complete": full.complete, "stop_reason": full.stop_reason, "n_rows": full.n_rows},
        "production_same_x_exit": {"complete": mid.complete, "stop_reason": mid.stop_reason, "n_rows": mid.n_rows},
        "limited": {"complete": lim.complete, "stop_reason": lim.stop_reason, "n_rows": lim.n_rows,
                    "stats": lim.limited_stats},
        "vs_production_same_x_exit": compare(mid),
        "vs_full_march": compare(full),
        "all_kept_nodes_incl_first_beyond_vs_production_same_x_exit": compare(mid, all_kept=True),
        "row_count_equal_to_production_same_x_exit": lim.n_rows == mid.n_rows,
        "dod_margins": dod_margins(lim, x_mid, extra_nodes=nxt),
    }
    res["passes"] = bool(res["vs_production_same_x_exit"]["bit_identical"] and res["vs_full_march"]["bit_identical"]
                         and res["row_count_equal_to_production_same_x_exit"] and lim.complete == mid.complete
                         and lim.limited_stats["exit_truncated_rows"] > 0)
    return res


def make_wall(theta_B, xb, rb, theta_E, Rt=1.0, Rd=0.382):
    from lrekit.components.nozzle.solvers.bounded import CompositeWall
    from lrekit.components.nozzle.solvers.nasa import ArcWall
    from lrekit.components.nozzle.solvers.wall import SplineWall
    arc = ArcWall(Rt, Rd, theta_B)
    bell = SplineWall(np.asarray(xb, float), np.asarray(rb, float), slope_start=math.tan(theta_B),
                      slope_end=math.tan(theta_E))
    return arc, bell, CompositeWall(arc, bell)


def replay_wall_limited(theta_B, xb, rb, theta_E, *, n_kernel, dtheta_deg, gamma=C.GAMMA, Rt=1.0, Rd=0.382,
                        Ru=1.5, x_exit=X_EXIT, max_rrcs=5000, keep_net=False, cf_ref_radius=1.0):
    """E07b harness: exit-plane-limited forward replay on arc(Rt, Rd, theta_B) + SplineWall(xb, rb)."""
    from lrekit.components.nozzle.solvers.bounded import crop_to_exit_plane, exit_plane_mass_flow
    from lrekit.components.nozzle.solvers.nasa import _make_throat_initial_line
    from tests.experiments.nozzle.moc_ext_e07_forward_replay import exit_plane_cf
    arc, bell, wall = make_wall(theta_B, xb, rb, theta_E, Rt, Rd)
    start = _make_throat_initial_line(Rt, Ru, theta_B, gamma, n_kernel, "kliegel_levine")
    t0 = time.time()
    net = march_limited(start, wall, gamma, arc=arc, dtheta_limit=math.radians(dtheta_deg), mass_tol=MASS_TOL,
                        on_invalid="stop", x_exit=x_exit, max_rrcs=max_rrcs)
    out = {"complete": bool(net.complete), "stop_reason": net.stop_reason, "n_rows": net.n_rows,
           "fold_location": net.fold_location, "limited_stats": net.limited_stats,
           "containment_violations": net.containment_violations, "truncated_rows": net.truncated_rows,
           "max_abs_mdot_rel_err_checked_rows": float(np.nanmax(np.abs(net.mdot_rel_err))),
           "march_wall_time_s": time.time() - t0, "arc_x_end": arc.x_end, "arc_r_end": arc.r_end,
           "wall": {k: np.array([getattr(row[0], k) for row in net.rows[1:]]) for k in ("x", "r", "M", "theta")},
           "mdot_start": float(net.massflow[0])}
    if net.complete:
        samples = crop_to_exit_plane(net, x_exit)
        cf, ex = exit_plane_cf(samples, gamma)
        out["cf"] = cf / (cf_ref_radius ** 2)
        out["exit_plane"] = ex
        out["mdot_exit"] = float(exit_plane_mass_flow(samples, gamma))
        out["mdot_exit_rel_to_start"] = (out["mdot_exit"] - out["mdot_start"]) / out["mdot_start"]
        out["dod_margins"] = dod_margins(net, x_exit)
    if keep_net:
        out["net"] = net
    return out


def run_level(run, report, level):
    from lrekit.components.nozzle.solvers.nasa import ArcWall, full_control_surface_thrust
    from lrekit.components.nozzle.solvers.variational import _control_surface_flow_nodes
    from tests.experiments.nozzle.moc_ext_e07_forward_replay import (LEVELS, RD_OVER_RT, RT, THRESH, bell_nodes_from_solution,
                                                    bvp_reference, compare_wall, mesh_subsample)
    run.add_input(C.PKL[level], report["artifact_sha256"][f"{level}_solution.pkl"])
    sol = C.load_solution(level)
    lv = LEVELS[level]
    kernel = sol.construction_diagnostics["bde_artifacts"]["kernel"]
    theta_B = float(kernel.theta_B)
    arc = ArcWall(RT, RD_OVER_RT * RT, theta_B)
    xb, rb, _ = bell_nodes_from_solution(sol, arc)
    rep = replay_wall_limited(theta_B, xb, rb, float(sol.theta_E), n_kernel=lv["n_kernel"], dtheta_deg=lv["dtheta_deg"],
                              keep_net=True)
    net = rep.pop("net")
    ref = bvp_reference(sol, C.GAMMA)
    fct = full_control_surface_thrust(kernel, _control_surface_flow_nodes(sol.control_surface), gamma=C.GAMMA,
                                      Rt=RT, pa_over_p0=0.0)
    cmp = compare_wall(rep["wall"], ref["wall"], arc.x_end)
    rrc, lrc = mesh_subsample(net)
    n_k = min(len(kernel.rrcs), net.n_rows)
    kernel_rows_identical = all(
        len(net.rows[j]) <= len(kernel.rrcs[j]) and all(p.M == q.M and p.x == q.x and p.r == q.r and
                                                        p.theta == q.theta
                                                        for p, q in zip(kernel.rrcs[j], net.rows[j]))
        for j in range(n_k))
    kernel_rows_truncated = sum(len(net.rows[j]) < len(kernel.rrcs[j]) for j in range(n_k))
    del net
    np.savez_compressed(run.out_dir / f"replay_{level}.npz",
                        fwd_wall_x=rep["wall"]["x"], fwd_wall_r=rep["wall"]["r"], fwd_wall_M=rep["wall"]["M"],
                        fwd_wall_theta=rep["wall"]["theta"], ref_wall_x=ref["wall"]["x"], ref_wall_M=ref["wall"]["M"],
                        ref_wall_theta=ref["wall"]["theta"], ref_cde=ref["cde"], bell_x=xb, bell_r=rb,
                        exit_r=rep.get("exit_plane", {}).get("r", np.zeros(0)),
                        exit_M=rep.get("exit_plane", {}).get("M", np.zeros(0)),
                        exit_theta=rep.get("exit_plane", {}).get("theta", np.zeros(0)),
                        cmp_x=cmp["x"], cmp_dM=cmp["dM"], cmp_dtheta_deg=cmp["dtheta_deg"],
                        **{f"rrc_{i}": a for i, a in enumerate(rrc)}, **{f"lrc_{i}": a for i, a in enumerate(lrc)})
    cf_rel = (rep["cf"] - ref["cf"]) / ref["cf"] if "cf" in rep else None
    out = {
        "level": level, "theta_B_deg": math.degrees(theta_B), "n_kernel": lv["n_kernel"], "dtheta_deg": lv["dtheta_deg"],
        "complete": rep["complete"], "stop_reason": rep["stop_reason"], "n_rows": rep["n_rows"],
        "fold_location": rep["fold_location"], "limited_stats": rep["limited_stats"],
        "containment_violations": rep["containment_violations"],
        "kernel_rows_prefix_identical_to_pickle": kernel_rows_identical,
        "kernel_rows_exit_truncated": int(kernel_rows_truncated), "kernel_rows": int(len(kernel.rrcs)),
        "max_abs_mdot_rel_err_checked_rows": rep["max_abs_mdot_rel_err_checked_rows"],
        "march_wall_time_s": rep["march_wall_time_s"],
        "cf_forward": rep.get("cf"), "cf_bvp": ref["cf"], "cf_rel_diff": cf_rel,
        "mdot_start": rep["mdot_start"], "mdot_exit": rep.get("mdot_exit"),
        "mdot_exit_rel_to_start": rep.get("mdot_exit_rel_to_start"),
        "mdot_bvp_cde": float(fct.mass_flux_cde), "mdot_bvp_kernel_throat": float(fct.kernel_throat_mass_flux),
        "mdot_exit_rel_to_bvp_cde": ((rep["mdot_exit"] - fct.mass_flux_cde) / fct.mass_flux_cde
                                     if "mdot_exit" in rep else None),
        "exit_plane_mass_averaged_p_over_p0": rep.get("exit_plane", {}).get("mass_averaged_p_over_p0"),
        "exit_plane_area_averaged_p_over_p0": rep.get("exit_plane", {}).get("area_averaged_p_over_p0"),
        "wall_compare": {k: v for k, v in cmp.items() if not isinstance(v, np.ndarray)},
        "dod_margins": rep.get("dod_margins"),
        "threshold_checks": ({"mach": cmp["mach_max_abs"] <= THRESH["mach_max_abs"],
                              "theta": cmp["theta_max_abs_deg"] <= THRESH["theta_max_abs_deg"],
                              "cf": abs(cf_rel) <= THRESH["cf_rel"]}
                             if rep["complete"] and cmp["mach_max_abs"] is not None else None),
    }
    np.savez_compressed(run.out_dir / f"forward_wall_{level}.npz", x=rep["wall"]["x"], r=rep["wall"]["r"],
                        M=rep["wall"]["M"], theta=rep["wall"]["theta"])
    return out


def main():
    from tests.experiments.nozzle.moc_ext_e07_forward_replay import THRESH
    run = C.Run("E07b", __file__)
    report = C.load_report()
    hv = harness_verification()
    print("harness", hv["passes"], hv["vs_production_same_x_exit"], hv["vs_full_march"], hv["limited"]["stats"])
    levels = {}
    if hv["passes"]:
        for lvl in ("baseline", "refined"):
            levels[lvl] = run_level(run, report, lvl)
            v = levels[lvl]
            print(lvl, v["stop_reason"], v["n_rows"], v["cf_rel_diff"], v["wall_compare"].get("mach_max_abs"),
                  v["wall_compare"].get("theta_max_abs_deg"), v["threshold_checks"])
    if not hv["passes"]:
        status = "FAIL"
    else:
        ok = all(v["threshold_checks"] and all(v["threshold_checks"].values()) for v in levels.values())
        status = "PASS-DIAGNOSTIC" if ok else "FAIL-DIAGNOSTIC"
    figs = make_figure(run, levels) if levels else []
    with open(run.out_dir / "harness_verification.json", "w") as f:
        json.dump(C.jsonable(hv), f, indent=1)
    result = {
        "status": status,
        "labels": ["DIAGNOSTIC", "harness variant: exit-plane-limited march (not production)",
                   "E07 original status unchanged"],
        "evidence_class": "verification (code-to-code consistency of two repo algorithms on the same model and start line)",
        "harness_verification": hv,
        "configuration": {
            "harness": "march_limited: copy of lrekit.components.nozzle.solvers.bounded.march_bounded_net with (i) RRC truncated after its "
                       "first point with x > x_exit (kept for interpolation, no axis closure); (ii) fold/subsonic/"
                       "upstream audit only on cells whose four corners have x <= x_exit, beyond-exit cells counted; "
                       "(iii) rows closing on the axis at x <= x_exit unchanged",
            "declared_deviation": "an exit-truncated row may have 2 points (row_too_short guard 2 instead of 3 for such "
                                  "rows); the per-row 2 % mass check is applied only to rows that close on the axis; "
                                  "exit-truncated rows carry no full-RRC mass (counted as mass_unchecked_rows); a row "
                                  "following a truncated row that fails to pass x_exit before running out of parents "
                                  "stops the march as domain_of_dependence_violation",
            "harness_check": "E05 source-flow oracle (cone 15 deg, gamma 1.4, n 49; scripts/moc_ext_e05_source_flow_oracle"
                             ".build_case imported read-only), x_exit = mid-domain of the wall; every limited-march node "
                             "with x <= x_exit must equal the production march node (all CharPoint fields, exact) both "
                             "for production with the same x_exit and for the untruncated full march",
            "dod_check": f"theta+mu < 90 deg and theta-mu > -90 deg at retained nodes within {DOD_BAND} Rt of x_exit, at the "
                         "kept first-beyond nodes and (E05) at the next node of the untruncated march",
            "replay": "as E07: arc(Rt 1, Rd 0.382, theta_B) + SplineWall through the BVP wall nodes, KL start line "
                      "(Ru 1.5), dtheta 0.0625 / 0.03125 deg, mass_tol 0.02, on_invalid 'stop', x_exit 2.94",
            "diagnostic_thresholds": THRESH,
        },
        "metrics": {"levels": levels},
        "hypotheses": {"H_forward_replay_reproduces_bvp": ("supported" if status == "PASS-DIAGNOSTIC" else
                                                           ("undetermined" if not levels else "not supported"))},
        "figures": figs,
        "caption_facts": {
            "configuration": ("Exit-plane-limited forward bounded-net replay (script-local harness variant) of the "
                              "accepted t19 Nozzle-B walls (gamma 1.23; 32/48 and 48/64) with the BVP's KL start line, "
                              "throat arc and turning limits, compared with the BVP BDE wall states and C-D-E thrust."),
            "evidence_class": "verification (code-to-code consistency of two repo algorithms on the same model and start line); DIAGNOSTIC",
            "not_established": "Independence from the shared KL start line and unit processes; physical validity; "
                               "production behaviour of march_bounded_net (unchanged).",
        },
        "not_established": "Independence from the shared KL start line and unit processes; physical validity.",
        "notes": ["E07 (unlimited march) remains FAIL-DIAGNOSTIC as recorded."],
    }
    run.finish(result, extra_scripts=[C.REPO / "tests/experiments/nozzle/moc_ext_e07_forward_replay.py",
                                      C.REPO / "tests/experiments/nozzle/moc_ext_e05_source_flow_oracle.py"])
    print("E07b", status)


def make_figure(run, levels):
    plt = C.setup_style()
    from matplotlib.gridspec import GridSpec
    from tests.experiments.nozzle.moc_ext_e07_forward_replay import THRESH
    data = {lvl: dict(np.load(run.out_dir / f"replay_{lvl}.npz")) for lvl in levels}
    col = {"baseline": C.COLORS["teal"], "refined": C.COLORS["accent"]}
    lab = {"baseline": "32/48", "refined": "48/64"}
    fig = plt.figure(figsize=(C.FULL_W, 8.0))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1.0, 0.85, 0.9, 1.15], hspace=0.6, wspace=0.42,
                  left=0.12, right=0.98, top=0.975, bottom=0.06)
    ax = fig.add_subplot(gs[0, :])
    for lvl, d in data.items():
        ax.plot(d["fwd_wall_x"], d["fwd_wall_M"], color=col[lvl], lw=1.0, label=f"forward net {lab[lvl]}")
        ax.plot(d["ref_wall_x"], d["ref_wall_M"], ls="none", marker="o", ms=2.5, mfc="none", mec=col[lvl],
                label=f"BVP BDE wall {lab[lvl]}")
    ax.set_xlabel("x / R$_t$ (–)")
    ax.set_ylabel("wall Mach number (–)")
    ax.set_xlim(0, 3.0)
    ax.grid(True)
    ax.legend(loc="lower right", fontsize=6.5, ncol=2)
    C.panel_label(ax, "a")
    for c_, key, th, yl, letter in ((0, "cmp_dM", THRESH["mach_max_abs"], "ΔM forward − BVP (–)", "b"),
                                    (1, "cmp_dtheta_deg", THRESH["theta_max_abs_deg"], "Δθ forward − BVP (deg)", "c")):
        ax = fig.add_subplot(gs[1, c_])
        for lvl, d in data.items():
            if d["cmp_x"].size:
                ax.plot(d["cmp_x"], d[key], color=col[lvl], marker="o", ms=2.2, lw=0.9, label=lab[lvl])
        ax.axhspan(-th, th, color=C.COLORS["muted"], alpha=0.1, lw=0)
        ax.axhline(0, color=C.COLORS["muted"], lw=0.5)
        ax.set_xlabel("x / R$_t$ (–)")
        ax.set_ylabel(yl)
        ax.grid(True)
        ax.legend(loc="best", fontsize=6.5)
        C.panel_label(ax, letter)
    for c_, key, yl, letter, deg in ((0, "exit_M", "exit-plane Mach number (–)", "d", False),
                                     (1, "exit_theta", "exit-plane flow angle (deg)", "e", True)):
        ax = fig.add_subplot(gs[2, c_])
        for lvl, d in data.items():
            if d["exit_r"].size:
                ax.plot(d["exit_r"], np.degrees(d[key]) if deg else d[key], color=col[lvl], lw=1.0, label=lab[lvl])
        ax.set_xlabel("r / R$_t$ at x = 2.94 R$_t$ (–)")
        ax.set_ylabel(yl)
        ax.grid(True)
        ax.legend(loc="best", fontsize=6.5)
        C.panel_label(ax, letter)
    ax = fig.add_subplot(gs[3, :])
    lvl = "refined" if "refined" in data else next(iter(data))
    d = data[lvl]
    for k in sorted([k for k in d if k.startswith("rrc_")], key=lambda t: int(t[4:])):
        ax.plot(d[k][:, 0], d[k][:, 1], color="#AEB8BD", lw=0.3)
    for k in sorted([k for k in d if k.startswith("lrc_")], key=lambda t: int(t[4:])):
        ax.plot(d[k][:, 0], d[k][:, 1], color=C.COLORS["teal"], lw=0.3)
    ax.plot(d["fwd_wall_x"], d["fwd_wall_r"], color=C.COLORS["ink"], lw=1.1, label="wall")
    ax.plot(d["ref_cde"][:, 0], d["ref_cde"][:, 1], color=C.COLORS["accent"], lw=1.1, ls="--",
            label="BVP C-D-E surface")
    ax.plot([], [], color="#AEB8BD", lw=0.6, label="forward RRC (C−), subsampled")
    ax.plot([], [], color=C.COLORS["teal"], lw=0.6, label="forward LRC (C+), subsampled")
    ax.axvline(X_EXIT, color=C.COLORS["ink"], lw=0.7, ls=":")
    ax.set_xlim(-0.05, 3.15)
    ax.set_ylim(0, 3.0)
    ax.set_xlabel("x / R$_t$ (–)")
    ax.set_ylabel("r / R$_t$ (–)")
    ax.legend(loc="upper left", fontsize=6.3, ncol=2)
    C.panel_label(ax, "f")
    return [C.save_figure(fig, run.out_dir, "e07b_exit_limited_replay")]


if __name__ == "__main__":
    main()
