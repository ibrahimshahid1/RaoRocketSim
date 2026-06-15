"""
NASA kernel-march oracle parity (the kernel half of the J6 gate, landed early).

These tests pin the two root-cause fixes that unblocked the RRC march:

1. ``_visible_source_kl_throat`` honours the C++ *integer-division*
   semantics in ``z*(y*y - 5/8)`` (== ``z*y*y``; 5/8 is 0 in C++).  The
   binary that generated ``outputs_M3.5Perf`` ran with the term dropped,
   so binary fidelity requires dropping it too.  (The theory-correct
   ``y^2 - 5/8`` lives in ``raosim.transonic_kernel.kliegel_levine``.)
2. ``build_kernel`` evaluates the transonic start line with the
   *upstream* throat radius (C++ ``CalcInitialThroatLine(rUp, ...)``)
   via the new ``Ru=`` parameter, while the marching arc keeps the
   downstream radius.

Oracle: the checked-in NASA/JHU M3.5Perf outputs
(``Three-Dimensional-Nozzle-Design-Code-master/MOC_Grid_BDE/outputs_M3.5Perf``):
``TT'.out`` (start line), ``TT'BF_Kernel.out`` (full kernel grid),
``LastKernel.out`` (BD, j=57).

Historical note: before these fixes the march never advanced past TT'
(``rrcs == 1`` for every tested configuration) and silently fell back to
a degenerate arc+sonic-line BD — the actual root cause of the Phase 6
``max_scaled ~ 8`` convergence xfail (see JAX_DIFFERENTIABLE_PLAN.md §10
and tests/test_jax_convergence.py).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from raosim.moc import CharPoint
from raosim.gas_dynamics import prandtl_meyer, mach_angle
from raosim.nasa_moc import (
    ArcWall,
    _make_throat_initial_line,
    _rrc_march_step,
    _visible_source_kl_throat,
    build_kernel,
)

OUT = (Path(__file__).resolve().parent.parent
       / "Three-Dimensional-Nozzle-Design-Code-master" / "MOC_Grid_BDE"
       / "outputs_M3.5Perf")

pytestmark = pytest.mark.skipif(
    not OUT.exists(), reason="NASA M3.5Perf reference outputs not present",
)

GAMMA = 1.4
THETA_B = math.radians(15.2196)   # converged value from summary.out


def _read_ttprime():
    rows = []
    with open(OUT / "TT'.out") as f:
        next(f)
        for line in f:
            p = line.split()
            if len(p) >= 5:
                rows.append((float(p[1]), float(p[2]), float(p[3]),
                             float(p[4])))
    return rows  # wall-first: (x, r, M, theta_deg)


def _read_grid(name):
    rows: dict[int, list] = {}
    with open(OUT / name) as f:
        next(f)
        for line in f:
            p = line.split()
            if len(p) >= 6:
                i, j = int(p[0]), int(p[1])
                rows.setdefault(j, []).append(
                    (i, float(p[2]), float(p[3]), float(p[4]),
                     math.radians(float(p[5])))
                )
    return {j: sorted(v) for j, v in rows.items()}


def _to_char_point(x, r, M, theta_rad):
    nu = prandtl_meyer(M, GAMMA)
    return CharPoint(x=x, r=r, theta=theta_rad, M=M, nu=nu,
                     mu=mach_angle(M), compat_plus=theta_rad + nu,
                     compat_minus=theta_rad - nu)


# --------------------------------------------------------------------------- #
# 1. visible-source KL: integer-division semantics                             #
# --------------------------------------------------------------------------- #
def test_visible_kl_axis_hits_nasa_value():
    """At NASA's TT' axis coordinates the visible KL must give M=1.5."""
    nasa = _read_ttprime()
    x_axis, _, M_axis, _ = nasa[-1]
    M, theta = _visible_source_kl_throat(0.0, x_axis, GAMMA, 1.0)
    assert M == pytest.approx(M_axis, abs=2e-4)   # 1.5000
    assert theta == pytest.approx(0.0, abs=1e-12)


def test_throat_initial_line_matches_fixture():
    """TT' from the source-faithful mode matches TT'.out point for point."""
    nasa = _read_ttprime()
    tt = _make_throat_initial_line(
        1.0, 1.0, THETA_B, GAMMA, len(nasa), "nasa_visible_kliegel_levine",
    )
    wall_first = list(reversed(tt))
    assert len(wall_first) == len(nasa)
    for w, (nx, nr, nM, nth_deg) in zip(wall_first, nasa):
        assert w.x == pytest.approx(nx, abs=5e-6)
        assert w.r == pytest.approx(nr, abs=5e-6)
        assert w.M == pytest.approx(nM, abs=5e-5)
        assert math.degrees(w.theta) == pytest.approx(nth_deg, abs=5e-3)


# --------------------------------------------------------------------------- #
# 2. unit-process row march reproduces the NASA grid                           #
# --------------------------------------------------------------------------- #
def test_march_step_reproduces_nasa_row_1_from_row_0():
    """Feed NASA's own j=0 row; Python's unit processes must emit NASA's j=1."""
    grid = _read_grid("TT'BF_Kernel.out")
    j0 = [_to_char_point(x, r, M, th) for _, x, r, M, th in grid[0]]
    arc = ArcWall(1.0, 1.0, THETA_B)
    new = _rrc_march_step(list(reversed(j0)), arc, GAMMA,
                          0.5 * math.pi / 180.0)
    assert new is not None, "march step failed from NASA's own row 0"
    py1 = list(reversed(new))             # wall-first
    nasa1 = grid[1]
    assert len(py1) == len(nasa1)
    for p, (_, nx, nr, nM, nth) in zip(py1, nasa1):
        assert p.x == pytest.approx(nx, abs=2e-5)
        assert p.r == pytest.approx(nr, abs=2e-5)
        assert p.M == pytest.approx(nM, abs=2e-4)
        assert p.theta == pytest.approx(nth, abs=2e-4)


def test_build_kernel_marches_and_matches_last_kernel():
    """End-to-end: 58 rows and BD == LastKernel.out within tight tolerance."""
    grid_last = _read_grid("LastKernel.out")
    nasa_bd = grid_last[57]
    k = build_kernel(1.0, 1.0, THETA_B, GAMMA, 101,
                     starting_line_method="nasa_visible_kliegel_levine")
    assert len(k.rrcs) == 58, (
        f"expected NASA's 58 kernel rows (j=0..57), got {len(k.rrcs)}"
    )
    bd = k.bd  # wall-first
    assert len(bd) == len(nasa_bd)
    # B (wall end) and axis end, plus every node within tolerance.
    for node, (_, nx, nr, nM, nth) in zip(bd, nasa_bd):
        assert node.x == pytest.approx(nx, abs=5e-4)
        assert node.r == pytest.approx(nr, abs=5e-4)
        assert node.M == pytest.approx(nM, abs=2e-3)
        assert node.theta == pytest.approx(nth, abs=2e-3)


# --------------------------------------------------------------------------- #
# 3. the Rao-geometry regression that used to silently fall back               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["kliegel_levine",
                                  "nasa_visible_kliegel_levine"])
def test_rao_geometry_kernel_marches_with_upstream_radius(mode):
    """Rt=0.02 / Rd=0.382Rt / Ru=1.5Rt / theta_B=30 deg must march (rrcs >> 1).

    Before the Ru fix the KL line was evaluated with the *downstream*
    radius (0.382), far outside the series' validity, and the march died
    on its first interior point.
    """
    Rt = 0.020
    k = build_kernel(Rt, 0.382 * Rt, math.radians(30.0), GAMMA, 24,
                     starting_line_method=mode, Ru=1.5 * Rt)
    assert len(k.rrcs) > 30, (
        f"kernel march regressed: only {len(k.rrcs)} rows "
        "(the pre-fix failure mode was rrcs == 1)"
    )
    # Phase 12.4: the march must now reach the commanded wall angle.
    # (Historic behaviour: the mass-integral's dr <= 1e-15 guard zeroed
    # folded-RRC segments and tripped the 5% sanity check at theta_w ~
    # 24.1-24.4 deg, masquerading as a unit-process cap.)
    assert k.reached_wall, "kernel march no longer reaches theta_B"
    assert math.degrees(k.bd[0].theta) == pytest.approx(30.0, abs=1e-3)
    # ... and the BD descends from the throat-arc region to the axis
    # through supersonic states.
    assert k.bd[-1].r == pytest.approx(0.0, abs=1e-9)
    Ms = [n.M for n in k.bd]
    assert min(Ms) > 1.0
    assert max(Ms) > 2.0


# --------------------------------------------------------------------------- #
# 4. Phase 12.4 — folded RRCs and the mass integral                            #
# --------------------------------------------------------------------------- #
def test_folded_rrc_mass_integral_keeps_march_alive():
    """The ~24.2 deg "march cap" was the mass check, not the unit process.

    At Rao's sharp downstream radius (Rd = 0.382 Rt) the RRC slope
    tan(theta - mu) changes sign mid-row once the wall angle passes
    ~24.3 deg: theta crosses mu along the characteristic, so the row
    climbs in r before descending to the axis (a benign fold — verified
    non-crossing against neighbouring RRCs, hence not a limit line).
    The C++ integrates straight through such segments
    (fabs(mdot_a)*fabs(da), MOC_GridCalc_BDE.cpp:3217-3228); an earlier
    port revision zeroed them (dr <= 1e-15 guard), dropped ~13% of the
    row mass on the first folded RRC, and tripped build_kernel's 5%
    sanity check.  This pins the fixed behaviour.
    """
    from raosim.nasa_moc import calc_massflow_along_rrc

    Rt = 0.020
    k = build_kernel(Rt, 0.382 * Rt, math.radians(30.0), GAMMA, 60,
                     starting_line_method="kliegel_levine", Ru=1.5 * Rt)
    assert k.reached_wall

    # (a) the geometric feature is present: BD (wall-first) is
    #     non-monotone in r — it climbs somewhere mid-row.
    r = np.asarray([n.r for n in k.bd])
    climbs = np.diff(r) > 0.0
    assert climbs.any(), (
        "expected a folded BD at theta_B=30 deg / Rd=0.382Rt — if the "
        "fold is gone the flow solution changed, not just the integral"
    )
    # ... but it still spans wall to axis.
    assert r[0] > Rt
    assert r[-1] == pytest.approx(0.0, abs=1e-9)

    # (b) the cumulative grid is monotone: massflow is stored
    #     axis-zero / wall-max, so the wall-first array must be
    #     non-increasing (every segment contributes |flux| >= 0).
    mf = calc_massflow_along_rrc(k.bd, GAMMA)
    assert (np.diff(mf) <= 1e-15).all()

    # (c) total BD mass matches the start-line mass to well inside the
    #     5% sanity tolerance (observed: ~0.13% at this resolution).
    mdot_tt = float(k.massflow[0][0])
    mdot_bd = float(mf[0])
    assert abs(mdot_bd - mdot_tt) / mdot_tt < 5e-3


def test_march_past_cap_is_resolution_consistent():
    """Wall Mach at theta_B=30 deg agrees across march resolutions."""
    Rt = 0.020
    MB = []
    for nk in (24, 60):
        k = build_kernel(Rt, 0.382 * Rt, math.radians(30.0), GAMMA, nk,
                         starting_line_method="kliegel_levine", Ru=1.5 * Rt)
        assert k.reached_wall
        MB.append(k.bd[0].M)
    assert MB[0] == pytest.approx(MB[1], rel=2e-3)
