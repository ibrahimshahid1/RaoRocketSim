"""
Bounded characteristic net (raosim.moc_bounded) — topology and oracle gates.

These are the gates from docs/MOC_MARCH_TOPOLOGY_REMEDIATION_2026-09-21.md
Phase 0/2.  They would all FAIL on the deprecated front march
(``raosim.moc.march_coupled_net``): its axis/wall points were never
parents, the net shrank by one point per row and died at ~30 % of the
nozzle, and its exit plane was a two-point interpolation.

Oracle B uses the NASA/JHU M3.5 perfect-nozzle run shipped in
``Three-Dimensional-Nozzle-Design-Code-master/MOC_Grid_BDE/outputs_M3.5Perf``:
the wall from ``wall.out`` (throat arc Rd/R* = 1 up to theta_B = 15.2196 deg,
then the BFE contour), the NASA-visible Kliegel–Levine start line, and the
exit-plane table printed at the end of ``summary.out`` (stream thrust
2518.05 lbf, mass flow 35.9981 lbm/s, 0.7537 % initial-vs-exit mass
difference, uniform M = 3.5).  Note ``summary.out``'s "2-D Gross Thrust
1971.99 lbf" is the THROAT-plane figure, not the exit.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from raosim.legacy_io import parse_wall_out
from raosim.moc import (
    UpstreamIntersection,
    _make_point,
    approximate_starting_line,
    march_coupled_net,
    sample_exit_plane,
    solve_flowfield,
    solve_interior_point,
    compute_exit_thrust,
)
from raosim.moc_bounded import (
    CompositeWall,
    crop_to_exit_plane,
    exit_plane_mass_flow,
    intersect_wall_strict,
    march_bounded_net,
)
from raosim.nasa_moc import (
    ArcWall,
    _make_throat_initial_line,
    build_kernel,
)
from raosim.nozzle_geometry import bell_nozzle_contour
from raosim.rao_variational import characteristic_net_links
from raosim.wall_model import SplineWall

NASA_DIR = (
    Path(__file__).resolve().parents[1]
    / "Three-Dimensional-Nozzle-Design-Code-master" / "MOC_Grid_BDE"
    / "outputs_M3.5Perf"
)
GAMMA_NASA = 1.4
THETA_B_NASA_DEG = 15.2196          # summary.out "Initial Expansion Angle"
NASA_EXIT_THRUST_LBF = 2518.05      # summary.out exit-plane Stream Thrust
NASA_EXIT_MDOT_LBM_S = 35.9981      # summary.out exit-plane Massflow
NASA_MDOT_DIFF_PCT = 0.753673       # summary.out "% diffence in initial and exit"
NASA_P0_PSIA, NASA_T0_R, NASA_MW = 500.0, 530.0, 28.96


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


def _bezier_wall(Rt=0.02, eps=10.0, gamma=1.2, length_pct=80.0):
    c = bell_nozzle_contour(Rt=Rt, epsilon=eps, length_pct=length_pct, gamma=gamma)
    xb, yb = np.asarray(c["x_bell"]), np.asarray(c["y_bell"])
    wall = SplineWall(
        xb, yb,
        slope_start=math.tan(math.radians(c["theta_n"])),
        slope_end=math.tan(math.radians(c["theta_e"])),
    )
    return wall, c


def _nasa_wall():
    nasa = parse_wall_out(NASA_DIR / "wall.out")
    cols = list(nasa.columns)
    x = np.asarray(nasa.column(cols[2]), dtype=float)
    r = np.asarray(nasa.column(cols[3]), dtype=float)
    M = np.asarray(nasa.column(cols[4]), dtype=float)
    theta_B = math.radians(THETA_B_NASA_DEG)
    arc = ArcWall(1.0, 1.0, theta_B)
    iB = int(np.argmin(np.abs(x - arc.x_end)))
    xb, rb = x[iB:].copy(), r[iB:].copy()
    xb[0], rb[0] = arc.x_end, arc.r_end
    bell = SplineWall(xb, rb, slope_start=math.tan(theta_B), slope_end=0.0)
    return arc, bell, CompositeWall(arc, bell), (x, r, M)


@pytest.fixture(scope="module")
def nasa_net():
    if not (NASA_DIR / "wall.out").exists():
        pytest.skip("NASA M3.5Perf oracle outputs not present")
    arc, bell, comp, wall_out = _nasa_wall()
    start = _make_throat_initial_line(
        1.0, 1.0, arc.theta_max, GAMMA_NASA, 101, "nasa_visible_kliegel_levine",
    )
    net = march_bounded_net(
        start, comp, GAMMA_NASA, arc=arc,
        dtheta_limit=math.radians(0.5), mass_tol=0.02, on_invalid="stop",
    )
    return arc, bell, comp, wall_out, net


# --------------------------------------------------------------------------- #
# Oracle A — kernel identity                                                  #
# --------------------------------------------------------------------------- #


def test_kernel_rows_are_bit_identical_to_build_kernel(nasa_net):
    """Rows on the throat arc reuse the NASA arc unit processes verbatim."""
    arc, _, _, _, net = nasa_net
    kernel = build_kernel(
        1.0, 1.0, arc.theta_max, GAMMA_NASA, n_kernel=101,
        starting_line_method="nasa_visible_kliegel_levine",
        dtheta_limit=math.radians(0.5),
    )
    assert kernel.reached_wall
    n = len(kernel.rrcs)
    assert net.n_rows > n
    for j in range(n):
        a, b = kernel.rrcs[j], net.rows[j]
        assert len(a) == len(b), f"row {j}: {len(a)} vs {len(b)}"
        assert max(abs(p.M - q.M) for p, q in zip(a, b)) == 0.0
        assert max(abs(p.x - q.x) for p, q in zip(a, b)) == 0.0


# --------------------------------------------------------------------------- #
# Topology gates (Phase 0.4 i–v)                                              #
# --------------------------------------------------------------------------- #


def _parent_usage(net):
    """(#rows whose wall point is a parent, #rows whose axis point is a parent)."""
    wall_used = set()
    axis_used = set()
    for j in range(1, net.n_rows):
        pmap = net.parents[j]
        for entry in pmap:
            for fam in ("cplus", "cminus"):
                src = entry.get(fam)
                if src is None:
                    continue
                prow, pidx = src
                if pidx == 0:
                    wall_used.add(prow)
                if pidx == len(net.rows[prow]) - 1 and abs(net.rows[prow][-1].r) <= 1e-12:
                    axis_used.add(prow)
    return wall_used, axis_used


def test_boundary_points_are_parents(nasa_net):
    """(i) Every wall point and every axis point feeds the next RRC."""
    _, _, _, _, net = nasa_net
    wall_used, axis_used = _parent_usage(net)
    # every row's wall point except the last row's is the C- parent of [1][j]
    assert all(j in wall_used for j in range(1, net.n_rows - 1))
    # every axis point (except the last row's) is a C+ parent of the next row
    axis_rows = [j for j in range(1, net.n_rows - 1) if abs(net.rows[j][-1].r) <= 1e-12]
    assert axis_rows and all(j in axis_used for j in axis_rows)


def test_point_count_never_shrinks_except_axis_truncation(nasa_net):
    """(ii) iLast[j] = iLast[j-1] + specialFlag, reduced only by r<0 truncation."""
    _, _, _, _, net = nasa_net
    shrink_events = 0
    for j in range(1, net.n_rows):
        expected = len(net.rows[j - 1]) + (1 if net.special[j] else 0)
        assert len(net.rows[j]) <= expected
        if len(net.rows[j]) < expected:
            shrink_events += 1
    # NASA's own M3.5Perf run truncates RRC 1 (101 -> 94 points) at the
    # axis and then never shrinks again (TT'BF_Kernel.out: 101, 94 x 17,
    # 95 ... 121); every shrink must be an r<0 truncation, never a dropped
    # boundary point.
    assert shrink_events == net.truncated_rows
    assert net.truncated_rows <= 2
    assert len(net.rows[1]) == 94


def test_interior_points_lie_inside_the_wall(nasa_net):
    """(iii) no computed point above the wall."""
    _, _, comp, _, net = nasa_net
    assert net.containment_violations == 0
    for row in net.rows:
        for p in row[1:]:
            if p.x <= comp.x_end:
                assert p.r <= comp.r(p.x) + 1e-9


def test_net_reaches_the_exit_with_distinct_wall_points(nasa_net):
    """(iv) the march terminates at the exit lip, wall points all distinct."""
    _, bell, _, _, net = nasa_net
    assert net.complete and net.stop_reason == "exit_reached"
    assert net.wall_x[-1] == pytest.approx(bell.x_end, abs=1e-9)
    assert len(set(round(x, 9) for x in net.wall_x)) == len(net.wall_x)
    assert all(np.diff(net.wall_x) > 0)


def test_mass_flow_conserved_along_every_rrc(nasa_net):
    """(v) NASA's 2 % per-RRC rule; measured ≤ 0.8 % on M3.5Perf."""
    _, _, _, _, net = nasa_net
    assert max(abs(e) for e in net.mdot_rel_err) < 0.01


# --------------------------------------------------------------------------- #
# Oracle B — NASA M3.5 perfect nozzle                                         #
# --------------------------------------------------------------------------- #


def test_oracle_b_wall_mach_matches_wall_out(nasa_net):
    arc, _, _, (x, r, M), net = nasa_net
    wx = np.array([row[0].x for row in net.rows[1:]])
    wM = np.array([row[0].M for row in net.rows[1:]])
    sel = wx > arc.x_end
    assert sel.sum() > 50
    d = wM[sel] - np.interp(wx[sel], x, M)
    assert float(np.sqrt(np.mean(d ** 2))) < 1e-3      # measured 3.1e-4
    assert float(np.max(np.abs(d))) < 5e-3            # measured 1.3e-3


def test_oracle_b_axis_mach_matches_center_out(nasa_net):
    _, _, _, _, net = nasa_net
    c = np.loadtxt(NASA_DIR / "center.out", skiprows=2)
    cx, cM = c[:, 1], c[:, 3]
    # center.out row 0 is the M=1.5 KL cap point at x=0.724, upstream of
    # rows 1+; compare only where x is monotone (x >= 1).
    order = np.argsort(cx)
    cx, cM = cx[order], cM[order]
    ax = np.array(sorted(
        (row[-1].x, row[-1].M) for row in net.rows if abs(row[-1].r) <= 1e-12
    ))
    sel = (ax[:, 0] >= 1.0) & (ax[:, 0] <= cx.max())
    assert sel.sum() > 100
    d = ax[sel, 1] - np.interp(ax[sel, 0], cx, cM)
    # NASA builds the B-D-E strip BACKWARD from the uniform DE line, so its
    # centreline is uniform M=3.5 by construction from x~9.5; the forward
    # analysis of the same wall overshoots ~1 % where the reflected waves
    # focus on the axis.  Measured at DTHETAB=0.5 deg: RMS 8.9e-3, max
    # 3.6e-2; at 0.25 deg: 4.8e-3 / 2.1e-2 (discretisation, halves with
    # the step).
    assert float(np.sqrt(np.mean(d ** 2))) < 1.5e-2
    assert float(np.max(np.abs(d))) < 5e-2


def test_oracle_b_exit_plane_is_uniform_mach_3p5(nasa_net):
    """A perfect nozzle must deliver uniform, axial M = 3.5 at the exit."""
    _, _, _, _, net = nasa_net
    ex = crop_to_exit_plane(net)
    assert len(ex) >= 30
    Ms = np.array([s["M"] for s in ex])
    th = np.degrees([s["theta"] for s in ex])
    assert Ms.mean() == pytest.approx(3.5, abs=5e-3)   # measured 3.5009
    assert Ms.std() < 1e-2                              # measured 1.7e-3
    assert float(np.sqrt(np.mean(th ** 2))) < 0.05      # measured 0.003 deg
    assert ex[0]["r"] == 0.0 and ex[0]["theta"] == 0.0


def test_oracle_b_exit_thrust_and_mass_flow_match_summary_out(nasa_net):
    _, _, _, _, net = nasa_net
    ex = crop_to_exit_plane(net)
    P0 = NASA_P0_PSIA * 6894.757
    T0 = NASA_T0_R / 1.8
    Rg = 8314.46 / NASA_MW
    Rt_m = 0.0254
    ex_d = [{**s, "r": s["r"] * Rt_m} for s in ex]
    em = compute_exit_thrust(ex_d, GAMMA_NASA, p_ambient=0.0, Pc=P0, Tc=T0,
                             R_gas=Rg, throat_radius=Rt_m)
    thrust_lbf = em["F"] / 4.448222
    assert thrust_lbf == pytest.approx(NASA_EXIT_THRUST_LBF, rel=2e-3)  # −0.013 %
    rho0, a0 = P0 / (Rg * T0), math.sqrt(Rg * T0)
    mdot_lbm = exit_plane_mass_flow(ex, GAMMA_NASA) * rho0 * a0 * Rt_m ** 2 / 0.45359237
    assert mdot_lbm == pytest.approx(NASA_EXIT_MDOT_LBM_S, rel=2e-3)   # −0.025 %
    drift_pct = 100.0 * (mdot_lbm / (net.massflow[0] * rho0 * a0 * Rt_m ** 2 / 0.45359237) - 1.0)
    assert drift_pct == pytest.approx(NASA_MDOT_DIFF_PCT, abs=0.1)


# --------------------------------------------------------------------------- #
# solve_flowfield on a real wall + coalescence honesty                       #
# --------------------------------------------------------------------------- #


def test_solve_flowfield_reports_bezier_compression_coalescence():
    """The chart Bézier is not Rao's shock-free contour: its compression
    waves coalesce ~20 mm downstream of N (Rt=20 mm, eps=10).  The march
    must stop there and say so — never integrate through it.  The fold
    location converges with resolution (physics, not discretisation)."""
    wall, _ = _bezier_wall()
    locs = []
    for n, dth in ((24, 0.5), (48, 0.25)):
        res = solve_flowfield(0.02, 10.0, 1.2, wall, n_char=n,
                              dtheta_limit_deg=dth, mass_tol=0.05)
        net = res["net"]
        assert res["complete"] is False
        assert res["stop_reason"] == "rrc_coalescence"
        assert net.fold_location is not None
        assert res["exit_metrics"]["complete"] is False
        assert net.containment_violations == 0
        locs.append(net.fold_location)
    (x1, r1), (x2, r2) = locs
    assert abs(x1 - x2) < 3e-3 and abs(r1 - r2) < 5e-4
    assert 0.015 < x1 < 0.035


def test_solve_flowfield_refuses_non_data_line_methods():
    wall, _ = _bezier_wall()
    for bad in ("area_ratio", "hall", "sauer_modified"):
        with pytest.raises(ValueError):
            solve_flowfield(0.02, 10.0, 1.2, wall, n_char=12,
                            starting_line_method=bad)


def test_char_rows_carry_explicit_parent_map(nasa_net):
    """CharRow.parents drives rao_variational.characteristic_net_links."""
    _, _, _, _, net = nasa_net
    rows = net.to_char_rows()
    assert rows[0].parents is None and all(r.parents is not None for r in rows[1:])
    links = characteristic_net_links(rows)
    # every non-first row's wall point is a C+ hit, axis point a C- closure
    assert len(links["wall"]) == net.n_rows - 1
    axis_rows = sum(1 for r in rows[1:] if r.axis is not None)
    assert len(links["axis"]) == axis_rows
    # the C- parent of the first interior point of row j is row j's wall point
    first_interior = [l for l in links["cminus"] if l.role == "interior"
                      and l.row == 5 and l.child is rows[5].interior[-1]]
    assert first_interior and first_interior[0].parent is rows[5].wall
    # the previous row's axis point is a C+ parent somewhere in row j
    axis_parents = [l for l in links["cplus"] if l.row == 5 and l.parent is rows[4].axis]
    assert axis_parents


def test_sample_exit_plane_matches_crop(nasa_net):
    _, bell, _, _, net = nasa_net
    rows = net.to_char_rows()
    raw = sample_exit_plane(rows, bell.x_end, GAMMA_NASA, n_samples=0)
    ref = crop_to_exit_plane(net)
    assert len(raw) == len(ref)
    assert max(abs(a["M"] - b["M"]) for a, b in zip(raw, ref)) < 1e-12
    resampled = sample_exit_plane(rows, bell.x_end, GAMMA_NASA, n_samples=25)
    assert len(resampled) == 25 and abs(np.mean([s["M"] for s in resampled]) - 3.5) < 5e-3


# --------------------------------------------------------------------------- #
# Unit-process policy + deprecated march                                      #
# --------------------------------------------------------------------------- #


def test_interior_point_upstream_policy():
    gamma = 1.4
    # Parents arranged so the C- (slope tan(theta-mu)) from above and the
    # C+ (slope tan(theta+mu)) from below meet UPSTREAM of both: with
    # mu ~ 72 deg at M = 1.05, theta_minus = +80 deg gives a C- rising at
    # +8 deg and theta_plus = -80 deg a C+ falling at -8 deg.
    p_minus = _make_point(0.010, 0.020, math.radians(80.0), 1.05, gamma)
    p_plus = _make_point(0.010, 0.010, math.radians(-80.0), 1.05, gamma)
    kept = solve_interior_point(p_minus, p_plus, gamma, on_upstream="keep")
    assert kept.x < 0.010
    with pytest.raises(UpstreamIntersection):
        solve_interior_point(p_minus, p_plus, gamma, on_upstream="raise")
    clamped = solve_interior_point(p_minus, p_plus, gamma, on_upstream="clamp")
    assert clamped.x >= 0.010
    with pytest.raises(ValueError):
        solve_interior_point(p_minus, p_plus, gamma, on_upstream="nope")


def test_front_march_is_deprecated_and_still_shrinks():
    """Regression pin of the OLD behaviour so the defect stays documented."""
    Rt, Rd = 0.02, 0.382 * 0.02
    theta_n = math.radians(30)
    Re = math.sqrt(10.0) * Rt
    Ln = (Re - Rt) / math.tan(math.radians(15)) * 0.8
    Ny = Rt + Rd * (1.0 - math.cos(theta_n))
    Nx = Rd * math.sin(theta_n)
    wall = SplineWall.from_controls(np.linspace(Ny, Re, 5)[1:-1], Nx, Ny, Ln, Re, theta_n)
    pts = approximate_starting_line(Rt, Rd, theta_n, 1.4, 10)
    with pytest.warns(DeprecationWarning):
        rows = march_coupled_net(pts, wall, 1.4, max_rows=8)
    lens = [len(r.all_points()) for r in rows]
    assert lens == [10, 10, 9, 8, 7, 6, 5, 4]
    assert rows[-1].wall.x < 0.05 * wall.x_end


def test_intersect_wall_strict_returns_none_past_exit():
    wall, _ = _bezier_wall()
    arc = ArcWall(0.02, 0.382 * 0.02, math.radians(30.0))
    comp = CompositeWall(arc, wall)
    # a shallow C+ from near the exit leaves through the exit plane
    assert intersect_wall_strict(comp, 0.12, 0.05, 0.05, x_lo=0.12) is None
    hit = intersect_wall_strict(comp, 0.05, 0.02, 1.0, x_lo=0.05)
    assert hit is not None and abs(hit[1] - comp.r(hit[0])) < 1e-9
