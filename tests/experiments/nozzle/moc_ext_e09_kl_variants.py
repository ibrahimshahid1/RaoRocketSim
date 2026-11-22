"""E09 (DIAGNOSTIC): KL start-line variants, v3 z*y^3 sign and M* -> M normalisation.

Opt-in experiment script (moc_extensions_plan_2026-11-21). Production coefficients are untouched: the variants are a
local copy of lrekit.components.nozzle.solvers.transonic.kliegel_levine, injected by monkeypatching the module attribute inside this
process only.
Run: .venv-jax/bin/python -m tests.experiments.nozzle.moc_ext_e09_kl_variants
"""

from __future__ import annotations

import json
import math

import numpy as np

from scripts import moc_ext_e00_common as C

VARIANTS = {
    "V0": {"sign": +1.0, "mach": "mstar", "desc": "repo series, M = hypot(U,V) (production)"},
    "V1": {"sign": -1.0, "mach": "mstar", "desc": "v3 z*y^3 sign negative, M = hypot(U,V)"},
    "V2": {"sign": +1.0, "mach": "true", "desc": "repo sign, M from M* (Anderson Eq. 3.37)"},
    "V3": {"sign": -1.0, "mach": "true", "desc": "sign negative and M* -> M"},
}
RC_LIST = (1.0, 2.0, 4.0, 8.0, 16.0)
GAMMAS = (1.2, 1.23, 1.4)
Y_SAMPLES = np.linspace(0.05, 0.95, 10)
DZ_SAMPLES = (-0.1, -0.05, 0.0, 0.05, 0.1)
FD_H = 1.0e-3
SLOPE_GAIN_MIN = 0.8
START_CASES = {"nozzle_b": {"Ru": 1.5, "gamma": 1.23, "n": 64}, "top": {"Ru": 1.5, "gamma": 1.2, "n": 48}}


def axi_coeffs(y, z, G, sign):
    y2 = y * y
    y3 = y2 * y
    y4 = y3 * y
    y5 = y4 * y
    y6 = y5 * y
    y7 = y6 * y
    z2 = z * z
    z3 = z2 * z
    u1 = 0.5 * y2 - 0.25 + z
    v1 = y3 / 4.0 - y / 4.0 + y * z
    u2 = ((2 * G + 9) * y4 / 24.0 - (4 * G + 15) * y2 / 24.0 + (10 * G + 57) / 288.0 + z * (y2 - 5.0 / 8.0)
          - (2 * G - 3) * z2 / 6.0)
    v2 = ((G + 3) * y5 / 9.0 - (20 * G + 63) * y3 / 96.0 + (28 * G + 93) * y / 288.0
          + z * ((2 * G + 9) * y3 / 6.0 - (4 * G + 15) * y / 12.0) + y * z2)
    u3 = ((556 * G * G + 1737 * G + 3069) * y6 / 10368.0 - (388 * G * G + 1161 * G + 1881) * y4 / 2304.0
          + (304 * G * G + 831 * G + 1242) * y2 / 1728.0 - (2708 * G * G + 7839 * G + 14211) / 82944.0
          + z * ((52 * G * G + 51 * G + 327) * y4 / 384.0 - (52 * G * G + 75 * G + 279) * y2 / 192.0
                 + (92 * G * G + 180 * G + 639) / 1152.0)
          + z2 * (-(7 * G - 3) * y2 / 8.0 + (13 * G - 27) / 48.0) + (4 * G * G - 57 * G + 27) * z3 / 144.0)
    v3 = ((6836 * G * G + 23031 * G + 30627) * y7 / 82944.0 - (3380 * G * G + 11391 * G + 15291) * y5 / 13824.0
          + (3424 * G * G + 11271 * G + 15228) * y3 / 13824.0 - (7100 * G * G + 22311 * G + 30249) * y / 82944.0
          + z * ((556 * G * G + 1737 * G + 3069) * y5 / 1728.0
                 + sign * (388 * G * G + 1161 * G + 1881) * y3 / 576.0
                 + (304 * G * G + 831 * G + 1242) * y / 864.0)
          + z2 * ((52 * G * G + 51 * G + 327) * y3 / 192.0 - (52 * G * G + 75 * G + 279) * y / 192.0)
          - z3 * (7 * G - 3) * y / 12.0)
    return u1, u2, u3, v1, v2, v3


def series_uv(y, x, G, Rc, sign):
    z = x * np.sqrt(2.0 * Rc / (G + 1.0))
    RSP = Rc + 1.0
    u1, u2, u3, v1, v2, v3 = axi_coeffs(y, z, G, sign)
    U = 1.0 + u1 / RSP + (u1 + u2) / (RSP ** 2) + (u1 + 2 * u2 + u3) / (RSP ** 3)
    V = math.sqrt((G + 1.0) / (2.0 * RSP)) * (v1 / RSP + (1.5 * v1 + v2) / (RSP ** 2)
                                              + (15.0 / 8.0 * v1 + 2.5 * v2 + v3) / (RSP ** 3))
    return U, V


def mach_from_mstar(ms, G):
    return math.sqrt(2.0 * ms * ms / ((G + 1.0) - (G - 1.0) * ms * ms))


def make_kl(sign, mach):
    from lrekit.core.gas_dynamics import isentropic_pressure_ratio
    from lrekit.components.nozzle.solvers.transonic import GEOM_AXI, TransonicState

    def kliegel_levine(r_over_Rt, x_over_Rt, gamma, Rc_over_Rt, geom=GEOM_AXI):
        if geom != GEOM_AXI:
            raise ValueError("E09 local copy implements the AXI branch only")
        if Rc_over_Rt <= 0.0:
            raise ValueError("Rc_over_Rt must be positive")
        G = float(gamma)
        y = float(r_over_Rt)
        z = float(x_over_Rt) * math.sqrt(2.0 * Rc_over_Rt / (G + 1.0))
        RSP = float(Rc_over_Rt) + 1.0
        u1, u2, u3, v1, v2, v3 = axi_coeffs(y, z, G, sign)
        U = 1.0 + u1 / RSP + (u1 + u2) / (RSP ** 2) + (u1 + 2 * u2 + u3) / (RSP ** 3)
        V = math.sqrt((G + 1.0) / (2.0 * RSP)) * (
            v1 / RSP + (1.5 * v1 + v2) / (RSP ** 2) + (15.0 / 8.0 * v1 + 2.5 * v2 + v3) / (RSP ** 3))
        if abs(V) < 1e-5:
            V = 0.0
        theta = math.atan2(V, U)
        if abs(theta) < 1e-5:
            theta = 0.0
        M = math.hypot(U, V)
        if mach == "true":
            M = mach_from_mstar(M, G)
        p_over_p0 = isentropic_pressure_ratio(M, G) if M > 1.0 else isentropic_pressure_ratio(max(M, 1.0 + 1e-9), G)
        return TransonicState(U, V, M, theta, p_over_p0)
    return kliegel_levine


class patched:
    def __init__(self, variant):
        self.v = VARIANTS[variant]

    def __enter__(self):
        import lrekit.components.nozzle.solvers.transonic as tk
        self.tk = tk
        self.orig = tk.kliegel_levine
        tk.kliegel_levine = make_kl(self.v["sign"], self.v["mach"])
        return self

    def __exit__(self, *exc):
        self.tk.kliegel_levine = self.orig


def identity_check():
    import lrekit.components.nozzle.solvers.transonic as tk
    local = make_kl(+1.0, "mstar")
    worst = 0.0
    n = 0
    for G in GAMMAS:
        for Rc in (0.382, 1.0, 1.5, 4.0):
            for y in np.linspace(0.0, 1.0, 11):
                for x in np.linspace(-0.2, 0.6, 9):
                    a = tk.kliegel_levine(y, x, G, Rc)
                    b = local(y, x, G, Rc)
                    worst = max(worst, max(abs(p - q) for p, q in zip(
                        (a.u_over_astar, a.v_over_astar, a.M, a.theta, a.p_over_p0),
                        (b.u_over_astar, b.v_over_astar, b.M, b.theta, b.p_over_p0))))
                    n += 1
    return {"points": n, "max_abs_diff": worst, "bit_identical": worst == 0.0}


def residuals(G, Rc, sign):
    h = FD_H
    w = np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / 12.0
    offs = np.array([-2, -1, 0, 1, 2], dtype=float)
    k = math.sqrt(2.0 * Rc / (G + 1.0))
    R_irr, R_con = [], []
    for y in Y_SAMPLES:
        zs = 0.25 - 0.5 * y * y
        for dz in DZ_SAMPLES:
            x = (zs + dz) / k
            U, V = series_uv(y, x, G, Rc, sign)
            Ux = sum(wi * series_uv(y, x + o * h, G, Rc, sign)[0] for wi, o in zip(w, offs)) / h
            Vx = sum(wi * series_uv(y, x + o * h, G, Rc, sign)[1] for wi, o in zip(w, offs)) / h
            Ur = sum(wi * series_uv(y + o * h, x, G, Rc, sign)[0] for wi, o in zip(w, offs)) / h
            Vr = sum(wi * series_uv(y + o * h, x, G, Rc, sign)[1] for wi, o in zip(w, offs)) / h
            a2 = 0.5 * (G + 1.0) - 0.5 * (G - 1.0) * (U * U + V * V)
            R_irr.append(Ur - Vx)
            R_con.append((a2 - U * U) * Ux - U * V * (Ur + Vx) + (a2 - V * V) * Vr + a2 * V / y)
    R_irr, R_con = np.array(R_irr), np.array(R_con)
    return {"irrot_rms": float(np.sqrt(np.mean(R_irr ** 2))), "cont_rms": float(np.sqrt(np.mean(R_con ** 2))),
            "irrot_max": float(np.max(np.abs(R_irr))), "cont_max": float(np.max(np.abs(R_con)))}


def fit_slope(eps, vals):
    A = np.vstack([np.log(eps), np.ones_like(eps)]).T
    s, _ = np.linalg.lstsq(A, np.log(vals), rcond=None)[0]
    return float(s)


def residual_study():
    eps = 1.0 / (1.0 + np.array(RC_LIST))
    out = {}
    for G in GAMMAS:
        out[str(G)] = {}
        for vname in ("V0", "V1"):
            rows = [residuals(G, Rc, VARIANTS[vname]["sign"]) for Rc in RC_LIST]
            rec = {"eps": eps, "rows": rows}
            for key in ("irrot_rms", "cont_rms"):
                vals = np.array([r[key] for r in rows])
                rec[f"slope_{key}"] = fit_slope(eps, vals)
                rec[f"slope_{key}_3smallest"] = fit_slope(eps[-3:], vals[-3:])
            out[str(G)][vname] = rec
        for key in ("irrot_rms", "cont_rms"):
            out[str(G)][f"gain_{key}"] = out[str(G)]["V1"][f"slope_{key}"] - out[str(G)]["V0"][f"slope_{key}"]
    return out


def start_line(case, variant):
    from lrekit.components.nozzle.solvers.nasa import MOCNode, _make_throat_initial_line, calc_massflow_along_rrc
    c = START_CASES[case]
    with patched(variant):
        pts = _make_throat_initial_line(1.0, c["Ru"], 0.5, c["gamma"], c["n"], "kliegel_levine")
    G = c["gamma"]
    wf = [MOCNode.from_char_point(p, G) for p in reversed(pts)]
    mdot = float(calc_massflow_along_rrc(wf, G)[0])
    rho_s = (2.0 / (G + 1.0)) ** (1.0 / (G - 1.0))
    a_s = math.sqrt(G * 2.0 / (G + 1.0))
    mdot_1d = math.pi * rho_s * a_s
    return {"x": np.array([p.x for p in pts]), "r": np.array([p.r for p in pts]), "M": np.array([p.M for p in pts]),
            "theta": np.array([p.theta for p in pts]), "mdot": mdot, "mdot_ideal_1d": mdot_1d,
            "mdot_over_ideal": mdot / mdot_1d}


def replay_variant(variant, sol, theta_B, xb, rb):
    from tests.experiments.nozzle.moc_ext_e07_forward_replay import replay_wall
    with patched(variant):
        rep = replay_wall(theta_B, xb, rb, float(sol.theta_E), n_kernel=64, dtheta_deg=0.03125, keep_net=True)
    net = rep.pop("net")
    kernel_rows = None
    for j, row in enumerate(net.rows):
        if row[0].x >= rep["arc_x_end"] - 1e-9 or row[0].theta >= theta_B - 1e-6:
            kernel_rows = j + 1
            break
    bd = net.rows[kernel_rows - 1] if kernel_rows else net.rows[-1]
    out = {"complete": rep["complete"], "stop_reason": rep["stop_reason"], "n_rows": rep["n_rows"],
           "fold_location": rep["fold_location"], "cf": rep.get("cf"), "kernel_rows": kernel_rows,
           "wall_x": rep["wall"]["x"], "wall_M": rep["wall"]["M"], "wall_theta": rep["wall"]["theta"],
           "bd_x": np.array([p.x for p in bd]), "bd_r": np.array([p.r for p in bd]),
           "bd_M": np.array([p.M for p in bd]), "bd_theta": np.array([p.theta for p in bd]),
           "B": {"x": float(bd[0].x), "r": float(bd[0].r), "M": float(bd[0].M)}, "march_wall_time_s": rep["march_wall_time_s"]}
    if rep["stop_reason"] == "invalid_cell":
        out["first_row_failure"] = failing_row_detail(net, theta_B)
    del net
    return out


def failing_row_detail(net, theta_B, gamma=C.GAMMA, dtheta_deg=0.03125):
    """Re-run the unit processes of the row after the last stored one and report the first failing point (read-only)."""
    from lrekit.components.nozzle.solvers.nasa import ArcWall, _calc_arc_wall_point_with_special, _calc_interior_mesh_point
    from tests.experiments.nozzle.moc_ext_e07_forward_replay import RD_OVER_RT, RT
    arc = ArcWall(RT, RD_OVER_RT * RT, theta_B)
    prev = net.rows[-1]
    if prev[0].x >= arc.x_end - 1e-12:
        return {"available": False, "reason": "previous row is not on the throat arc"}
    wpt, sflag = _calc_arc_wall_point_with_special(prev, arc, gamma, math.radians(dtheta_deg))
    new = [wpt]
    i_last_prev = len(prev) - 1
    for i in range(1, i_last_prev + (1 if sflag else 0)):
        pt, neg = _calc_interior_mesh_point(prev, new, i, sflag, gamma)
        if neg:
            return {"available": True, "row": len(net.rows), "i": i, "reason": "negative radius"}
        if pt is None or pt.M < 1.0001:
            q = prev[min(i if sflag else i + 1, i_last_prev)]
            return {"available": True, "row": len(net.rows), "i": i, "row_points": len(prev),
                    "reason": "interior unit process returned None" if pt is None else f"M {pt.M:.6f} < 1.0001",
                    "cplus_parent": {"x": float(q.x), "r": float(q.r), "M": float(q.M)}}
        if pt.x < new[i - 1].x - 1e-12 * max(abs(new[i - 1].x), 1.0):
            return {"available": True, "row": len(net.rows), "i": i, "reason": "C- intersection upstream of parent",
                    "x": float(pt.x), "r": float(pt.r)}
        new.append(pt)
    return {"available": True, "row": len(net.rows), "reason": "no interior failure reproduced (axis closure?)"}


def kernel_builder_check(variant, theta_B):
    """Production build_kernel (the BVP's kernel builder, no fold audit) under the variant start line."""
    from lrekit.components.nozzle.solvers.nasa import build_kernel
    with patched(variant):
        k = build_kernel(1.0, 0.382, theta_B, C.GAMMA, n_kernel=64, starting_line_method="kliegel_levine",
                         max_rrcs=2000, dtheta_limit=math.radians(0.03125), Ru=1.5)
    return {"reached_wall": bool(k.reached_wall), "rows": len(k.rrcs), "fallback_used": bool(k.fallback_used),
            "B_M": float(k.rrcs[-1][0].M) if k.reached_wall else None,
            "last_wall_x": float(k.rrcs[-1][0].x)}


def main():
    from lrekit.components.nozzle.solvers.nasa import ArcWall
    from tests.experiments.nozzle.moc_ext_e07_forward_replay import RD_OVER_RT, RT, bell_nodes_from_solution

    run = C.Run("E09", __file__)
    report = C.load_report()
    ident = identity_check()
    res = residual_study()
    h1 = {G: {k: res[G][f"gain_{k}"] >= SLOPE_GAIN_MIN for k in ("irrot_rms", "cont_rms")} for G in res}
    h1_supported = all(all(v.values()) for v in h1.values())
    print("identity", ident)
    for G in res:
        print(G, {k: round(res[G][f"gain_{k}"], 3) for k in ("irrot_rms", "cont_rms")},
              {v: (round(res[G][v]["slope_irrot_rms"], 3), round(res[G][v]["slope_cont_rms"], 3)) for v in ("V0", "V1")})

    starts = {case: {v: start_line(case, v) for v in VARIANTS} for case in START_CASES}
    start_metrics = {}
    for case, d in starts.items():
        start_metrics[case] = {}
        for v, s in d.items():
            base = d["V0"]
            same_r = np.allclose(s["r"], base["r"], rtol=0, atol=0)
            start_metrics[case][v] = {
                "mdot_over_ideal_1d": s["mdot_over_ideal"], "mdot_rel_to_V0": (s["mdot"] - base["mdot"]) / base["mdot"],
                "radii_identical_to_V0": bool(same_r),
                "max_abs_dM_vs_V0": float(np.max(np.abs(s["M"] - base["M"]))),
                "max_abs_dtheta_deg_vs_V0": float(np.degrees(np.max(np.abs(s["theta"] - base["theta"])))),
                "max_abs_dx_vs_V0": float(np.max(np.abs(s["x"] - base["x"]))),
                "wall_point": {"x": float(s["x"][-1]), "M": float(s["M"][-1]), "theta_deg": math.degrees(s["theta"][-1])},
                "axis_point": {"x": float(s["x"][0]), "M": float(s["M"][0])},
            }

    run.add_input(C.PKL["refined"], report["artifact_sha256"]["refined_solution.pkl"])
    sol = C.load_solution("refined")
    kernel = sol.construction_diagnostics["bde_artifacts"]["kernel"]
    theta_B = float(kernel.theta_B)
    arc = ArcWall(RT, RD_OVER_RT * RT, theta_B)
    xb, rb, _ = bell_nodes_from_solution(sol, arc)
    replays = {}
    for v in VARIANTS:
        replays[v] = replay_variant(v, sol, theta_B, xb, rb)
        print(v, replays[v]["stop_reason"], replays[v]["n_rows"], replays[v]["fold_location"], replays[v]["B"],
              replays[v].get("first_row_failure"))
    del sol
    kchecks = {v: kernel_builder_check(v, theta_B) for v in VARIANTS}
    print("build_kernel", kchecks)
    t19_band = float(report["refinement"]["metrics"]["mach_max_abs"])
    replay_metrics = {}
    base = replays["V0"]
    for v, r in replays.items():
        if r["wall_x"].size and base["wall_x"].size:
            xs = np.linspace(0.0, min(r["wall_x"].max(), base["wall_x"].max()), 400)
            dM = np.interp(xs, r["wall_x"], r["wall_M"]) - np.interp(xs, base["wall_x"], base["wall_M"])
            dM_max = float(np.max(np.abs(dM)))
        else:
            dM_max = None
        replay_metrics[v] = {
            "complete": r["complete"], "stop_reason": r["stop_reason"], "n_rows": r["n_rows"],
            "kernel_rows": r["kernel_rows"], "fold_location": r["fold_location"], "cf": r["cf"],
            "arc_wall_max_abs_dM_vs_V0_over_common_x": dM_max,
            "wall_x_max_reached": float(r["wall_x"].max()) if r["wall_x"].size else None,
            "reached_B": bool(r["kernel_rows"] is not None),
            "first_row_failure": r.get("first_row_failure"),
            "B_M": r["B"]["M"] if r["kernel_rows"] else None,
            "B_dM_vs_V0": (r["B"]["M"] - base["B"]["M"]) if (r["kernel_rows"] and base["kernel_rows"]) else None,
            "bd_points": int(len(r["bd_x"])), "march_wall_time_s": r["march_wall_time_s"],
        }
    with open(run.out_dir / "series_residuals.json", "w") as f:
        json.dump(C.jsonable(res), f, indent=1)
    np.savez_compressed(run.out_dir / "e09_arrays.npz",
                        **{f"start_{c}_{v}_{k}": starts[c][v][k] for c in starts for v in VARIANTS
                           for k in ("x", "r", "M", "theta")},
                        **{f"replay_{v}_{k}": replays[v][k] for v in VARIANTS
                           for k in ("wall_x", "wall_M", "wall_theta", "bd_x", "bd_r", "bd_M", "bd_theta")})
    figs = make_figure(run, res, starts, replays, t19_band, xb[0])
    any_complete = any(r["complete"] for r in replays.values())
    status = ("PASS-DIAGNOSTIC" if (ident["bit_identical"] and h1_supported) else "FAIL-DIAGNOSTIC")
    result = {
        "status": status,
        "status_parts": {
            "identity_V0_vs_production": "PASS" if ident["bit_identical"] else "FAIL",
            "residual_H1": "supported" if h1_supported else "not supported",
            "start_line_sensitivity": "report-only, complete",
            "replay_sensitivity": ("INCOMPLETE: no variant completes. V0 and V3 reach B and then fold downstream of "
                                   "the exit plane (as E07); V1 folds near the axis in RRC 4; V2 fails in the first "
                                   "RRC (interior unit process returns None near the axis). Only start-line-to-B wall "
                                   "differences are available; no CF") if not any_complete else "complete",
        },
        "evidence_class": "verification (series residual); neither (sensitivity); all DIAGNOSTIC",
        "label": "DIAGNOSTIC: local monkeypatched copy of kliegel_levine; production coefficients untouched",
        "configuration": {
            "variants": {k: v["desc"] for k, v in VARIANTS.items()},
            "mach_conversion": "M^2 = 2 M*^2 / ((g+1) - (g-1) M*^2) (Anderson Eq. 3.37 per spec)",
            "residual_test": {
                "irrotationality": "dU/dr - dV/dx (U, V = q/a*, physical r = y Rt, x)",
                "continuity": "(a^2-U^2)U_x - UV(U_r+V_x) + (a^2-V^2)V_r + a^2 V/r, a^2 = (g+1)/2 - (g-1)/2 (U^2+V^2) "
                              "(steady irrotational isentropic axisymmetric gas-dynamic equation; same form as the "
                              "orchestrator's blind derivation step 3; Hall's series is for irrotational perfect-gas "
                              "flow with velocities over the sonic speed, Sivells AEDC-TR-78-63 p. 10)",
                "fd": f"4th-order central differences, h = {FD_H} (physical units)",
                "samples": "10 radii y in [0.05, 0.95] x 5 offsets z - z_s1(y) in {-0.1,-0.05,0,0.05,0.1}, "
                           "z_s1 = 1/4 - y^2/2 (first-order sonic line), x = z / sqrt(2 Rc/(g+1)); 50 points",
                "Rc_over_Rt": list(RC_LIST), "gammas": list(GAMMAS), "eps": "1/(1+Rc)",
                "fit": "least squares slope of ln(RMS residual) vs ln(eps) over all five Rc (3-smallest-eps slope "
                       "reported as secondary)",
                "H1_rule": f"slope(V1) - slope(V0) >= {SLOPE_GAIN_MIN} for both residuals at all three gammas",
            },
            "start_lines": START_CASES,
            "start_line_builder": "nasa_moc._make_throat_initial_line(Rt=1, Ru, theta_B unused, g, n, 'kliegel_levine'), "
                                  "M_min = 1.05 applied to the variant's M",
            "mdot_ideal_1d": "pi rho* a* Rt^2 in the units of nasa_moc.calc_massflow_along_rrc (rho0 = 1, V/sqrt(R T0))",
            "replay": "E07 replay_wall on the 48/64 wall, n_kernel 64, dtheta 0.03125 deg, per variant",
        },
        "metrics": {
            "identity_check": ident,
            "residual_slopes": {G: {v: {k: res[G][v][k] for k in res[G][v] if k.startswith("slope")} for v in ("V0", "V1")}
                                for G in res},
            "residual_slope_gain": {G: {k: res[G][f"gain_{k}"] for k in ("irrot_rms", "cont_rms")} for G in res},
            "residual_tables_file": C.rel(run.out_dir / "series_residuals.json"),
            "start_lines": start_metrics,
            "replay": replay_metrics,
            "bvp_build_kernel_check": kchecks,
            "t19_refinement_mach_max_abs": t19_band,
        },
        "hypotheses": {
            "H1_residual_order_gain": "supported" if h1_supported else "not supported",
            "H1_detail": h1,
            "H1_vs_orchestrator_KL_verdict": "consistent (verdict: repo sign wrong; corrected sign is -)"
            if h1_supported else "CONFLICT with orchestrator KL verdict; both records kept",
            "H2_start_line_changes_measurable": (
                "measurable; not small at the 1e-3 level: Nozzle-B max |dtheta| / |dM| vs V0 = "
                + ", ".join(f"{v} {start_metrics['nozzle_b'][v]['max_abs_dtheta_deg_vs_V0']:.3f} deg / "
                            f"{start_metrics['nozzle_b'][v]['max_abs_dM_vs_V0']:.4f}" for v in ("V1", "V2", "V3"))
                + "; the largest changes are mid-radius (theta) and on the axis (M), not at the wall point"),
            "H2_replay_below_t19_band": "undetermined (replay incomplete: forward net folds downstream of exit)",
        },
        "figures": figs,
        "caption_facts": {
            "configuration": ("Axisymmetric KL third-order series (repo copy) with the v3 z*y^3 sign as coded (V0) or "
                              "negated (V1), and M = q/a* or converted to M (V2, V3): (a, b) residual RMS over 50 "
                              "near-sonic points vs eps = 1/(1+Rc/Rt); (c-f) Nozzle-B start line (Ru/Rt 1.5, gamma 1.23, "
                              "n = 64); (g) forward-replay wall Mach from the start line to B relative to V0."),
            "evidence_class": "verification (series residual); neither (sensitivity); all DIAGNOSTIC",
            "not_established": "The original KL-1969 table; whether a corrected branch passes the BVP gates (needs an "
                               "approved ~50 min re-solve); any bell-wall or CF sensitivity (replay folded).",
        },
        "not_established": "The original KL-1969 table; whether a corrected branch passes the BVP gates (needs an "
                           "approved ~50 min re-solve).",
        "notes": [
            "The residual part uses raw series values (no 1e-5 V/theta clipping); the start-line and replay parts use "
            "the clipped local copy, bit-identical to production for V0.",
            "M* -> M does not change U, V, so V2/V3 share V0/V1 residuals; it changes the start-line M and the M_min "
            "push threshold.",
            "Context only (not a criterion): cuffel1969 PDF 2 reports KL/ideal mass flow 0.982 vs measured 0.985 for "
            "its own throat; here the start-line integral is a different quantity (flux through the KL start line).",
            "The V1/V2 forward-net stops occur in the kernel region (start line to B). nasa_moc.build_kernel (the BVP "
            "kernel builder) has no fold audit; bvp_build_kernel_check reports whether it reaches B under each variant "
            "(n_kernel 64, 0.03125 deg, 2000-row budget). A full BVP gate check needs the approved re-solve.",
        ],
    }
    run.finish(result, extra_scripts=[C.REPO / "tests/experiments/nozzle/moc_ext_e07_forward_replay.py"])
    print("E09", status)


def make_figure(run, res, starts, replays, band, x_B):
    plt = C.setup_style()
    from matplotlib.gridspec import GridSpec
    col = {"V0": C.COLORS["ink"], "V1": C.COLORS["teal"], "V2": C.COLORS["accent"], "V3": C.COLORS["muted"]}
    ls_v = {"V0": "-", "V1": "-", "V2": "--", "V3": ":"}
    mk_g = {"1.2": "o", "1.23": "s", "1.4": "^"}
    fig = plt.figure(figsize=(C.FULL_W, 8.6))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[1.1, 1.0, 1.0, 1.0], hspace=0.62, wspace=0.34,
                  left=0.1, right=0.98, top=0.975, bottom=0.055)
    for idx, (key, lab, letter) in enumerate((("irrot_rms", "irrotationality residual RMS (–)", "a"),
                                             ("cont_rms", "continuity residual RMS (–)", "b"))):
        ax = fig.add_subplot(gs[0, idx])
        for G in res:
            for v in ("V0", "V1"):
                eps = np.asarray(res[G][v]["eps"])
                vals = [r[key] for r in res[G][v]["rows"]]
                ax.plot(eps, vals, color=col[v], marker=mk_g[G], ms=3.2, lw=0.9, ls="-",
                        label=f"{v}, γ = {G}")
        e = np.array([0.0588, 0.5])
        for pw, ls_ in ((3, (0, (4, 2))), (4, (0, (1, 1.5)))):
            v0 = res["1.23"]["V0" if pw == 3 else "V1"]["rows"][-1][key]
            ax.plot(e, v0 * 0.35 * (e / e[0]) ** pw, color=C.COLORS["muted"], lw=0.7, ls=ls_, label=f"∝ ε$^{pw}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("ε = 1 / (1 + R$_c$/R$_t$) (–)")
        ax.set_ylabel(lab)
        ax.grid(True, which="major")
        if idx == 0:
            ax.legend(loc="lower right", fontsize=5.6, ncol=2, handlelength=1.8, columnspacing=0.8)
        C.panel_label(ax, letter)

    d = starts["nozzle_b"]
    specs = ((1, 0, "theta", "start-line θ (deg)", "c", True), (1, 1, "M", "start-line M (–)", "d", False),
             (2, 0, "theta", "Δθ vs V0 (deg)", "e", True), (2, 1, "M", "ΔM vs V0 (–)", "f", False))
    for row, c_, key, lab, letter, deg in specs:
        ax = fig.add_subplot(gs[row, c_])
        for v in VARIANTS:
            y = d[v][key] if row == 1 else d[v][key] - d["V0"][key]
            y = np.degrees(y) if deg else y
            ax.plot(d[v]["r"], y, color=col[v], ls=ls_v[v], lw=1.0, label=f"{v}")
        ax.set_xlabel("r / R$_t$ on the start line (–)")
        ax.set_ylabel(lab)
        ax.grid(True)
        if letter == "c":
            ax.legend(loc="upper left", fontsize=6.5, ncol=2)
        C.panel_label(ax, letter)

    ax = fig.add_subplot(gs[3, :])
    base = replays["V0"]
    xs = np.linspace(0.0, float(x_B), 400)
    for v in ("V1", "V2", "V3"):
        r = replays[v]
        if not r["wall_x"].size:
            ax.plot([], [], color=col[v], ls=ls_v[v], lw=1.0, label=f"{v}: no RRC completed")
            continue
        m = xs <= r["wall_x"].max()
        dM = np.interp(xs[m], r["wall_x"], r["wall_M"]) - np.interp(xs[m], base["wall_x"], base["wall_M"])
        reach = "" if r["kernel_rows"] else f" (stopped at x = {r['wall_x'].max():.3g})"
        ax.plot(xs[m], dM, color=col[v], ls=ls_v[v], lw=1.0, label=f"{v} − V0{reach}")
    ax.axhspan(-band, band, color=C.COLORS["teal"], alpha=0.12, lw=0, label="t19 refinement max |ΔM|")
    ax.axhline(0, color=C.COLORS["muted"], lw=0.5)
    ax.set_xlabel("x / R$_t$ along the throat arc, start line to B (–)")
    ax.set_ylabel("replayed wall ΔM (–)")
    ax.grid(True)
    ax.legend(loc="best", fontsize=6.3, ncol=2)
    C.panel_label(ax, "g")
    return [C.save_figure(fig, run.out_dir, "e09_kl_variants")]


if __name__ == "__main__":
    main()
