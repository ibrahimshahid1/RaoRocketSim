"""Phase 12.5 (REWRITE_PLAN §11.6): regression tests against NASA ground truth.

The NASA/JHU ``outputs_M3.5Perf`` grids are the canonical reference
implementation.  ``run_python_port_for_case`` reproduces the perfect-nozzle
workflow end to end with the Python ports — kernel march (KLThroat with
binary-faithful semantics), perfect-branch D/E placement (D = axis end of
BD, uniform-state DE per the C++ FindPointE PERFECT-AXI closed form), and
the BDE-region wall (``calc_bde_region``/``CalcWallContour``).

Measured agreement at first light: wall r(x) RMS = 1.8e-4 (max 4.8e-4)
against ``wall.out`` — already inside the plan's eventual 1e-3 target, so
the gate asserts 1e-3 directly rather than the loose initial 1e-2.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from raosim.legacy_io import parse_wall_out
from raosim.nasa_moc import (MOCNode, RaoTopology, build_kernel,
                             calc_bde_region, calc_massflow_along_rrc)

REPO_ROOT = Path(__file__).resolve().parent.parent
GAMMA = 1.4

CASES = {
    "M3.5Perf": dict(theta_b_deg=15.2196, Rt=1.0, Rd=1.0, n_kernel=101),
}


def _nasa_dir(case: str) -> Path:
    return (REPO_ROOT / "Three-Dimensional-Nozzle-Design-Code-master"
            / "MOC_Grid_BDE" / f"outputs_{case}")


pytestmark = pytest.mark.skipif(
    not _nasa_dir("M3.5Perf").exists(),
    reason="NASA reference outputs not present",
)


def run_python_port_for_case(case: str):
    """Kernel -> perfect topology -> BDE wall, all Python ports."""
    p = CASES[case]
    kernel = build_kernel(
        p["Rt"], p["Rd"], math.radians(p["theta_b_deg"]), GAMMA,
        p["n_kernel"], starting_line_method="nasa_visible_kliegel_levine",
    )
    bd = kernel.bd
    mdot = float(calc_massflow_along_rrc(bd, GAMMA)[0])
    D = bd[-1]                       # axis end (PERFECT branch)
    rE = math.sqrt(mdot / (math.pi * D.rho * D.u))
    xE = D.x + rE / math.tan(D.mu)   # uniform state: dr/dx = tan(mu_D)
    de = [MOCNode(D.x + (xE - D.x) * t, rE * t, max(D.M, 1.000001), 0.0,
                  GAMMA) for t in np.linspace(0.0, 1.0, 40)]
    topo = RaoTopology(
        B=bd[0].to_flow_node(), BD=tuple(n.to_flow_node() for n in bd),
        D=D.to_flow_node(), DE=tuple(n.to_flow_node() for n in de),
        E=de[-1].to_flow_node(), d_fraction=1.0,
        mass_BD=mdot, mass_DE=mdot, thrust_coefficient=float("nan"),
        theta_control=0.0, theta_B=float(kernel.theta_B),
        rao_stationarity_residual=float("nan"),
    )
    bfe = calc_bde_region(kernel, topo)
    wall = np.asarray(
        [(row[0].x, row[0].r) for row in kernel.rrcs if row]
        + [(pt.x, pt.r) for pt in bfe.wall_contour], dtype=float,
    )
    return kernel, topo, bfe, wall


@pytest.mark.parametrize("case", ["M3.5Perf"])
def test_nasa_wall_match(case):
    """Wall r(x) agrees with NASA wall.out to 1e-3 RMS (measured 1.8e-4)."""
    kernel, topo, bfe, wall = run_python_port_for_case(case)
    assert kernel.reached_wall and not kernel.fallback_used
    assert bfe.complete_remaining_mesh and bfe.wall_contour_complete

    nasa = parse_wall_out(_nasa_dir(case) / "wall.out")
    xcol = [c for c in nasa.columns if "x" in c.lower()][0]
    rcol = [c for c in nasa.columns if c.lower().startswith("r")][0]
    nx, nr = nasa.column(xcol), nasa.column(rcol)

    xc = np.linspace(max(nx.min(), wall[:, 0].min()),
                     min(nx.max(), wall[:, 0].max()), 200)
    ri = np.interp(xc, wall[:, 0], wall[:, 1])
    ni = np.interp(xc, nx, nr)
    rms = float(np.sqrt(np.mean((ri - ni) ** 2)))
    assert rms < 1e-3, f"wall r RMS vs NASA = {rms:.3e} (gate 1e-3)"
    # Exit point and length close to the published summary values.
    assert wall[-1, 0] == pytest.approx(12.5363, abs=2e-3)   # L/R*
    assert wall[-1, 1] == pytest.approx(2.5955, abs=2e-3)    # Re/R*


@pytest.mark.parametrize("case", ["M3.5Perf"])
def test_nasa_kernel_mass_consistency(case):
    """Full-BD mass equals the TT' throat mass within NASA's own 5% drift."""
    kernel, *_ = run_python_port_for_case(case)
    m0 = float(calc_massflow_along_rrc(kernel.rrcs[0], GAMMA)[0])
    mN = float(calc_massflow_along_rrc(kernel.bd, GAMMA)[0])
    assert mN == pytest.approx(m0, rel=5e-2)


# --------------------------------------------------------------------------- #
# Phase 12.4 follow-through: the fixed-end topology closes at the Rao          #
# reference design point once the kernel march clears the old ~24.2 deg cap.   #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_fixed_end_topology_closes_at_rao_reference():
    """set_theta_b must close (L, r_E) exactly for epsilon=10 / L80.

    Pre-12.4 the theta_B secant ran into the march's false mass-check
    halt at ~24.2 deg and the fixed-(L, epsilon) topology came out ~9%
    long.  With folded-RRC mass integration fixed, the secant converges
    at theta_B ~ 25.5 deg with the DE endpoint exactly on the commanded
    exit (observed: |dL/L| ~ 1e-6, |dr_E/r_E| ~ 1e-7, mass_BD == mass_DE).

    The wall built from this topology (kernel arc wall + BDE-region
    march) is a true bell: slope peaks ~ theta_B just after the throat
    arc and decreases monotonically to the exit (observed 26.3 deg peak
    at ~5% length, 11.2 deg exit at n_kernel=48).
    """
    from raosim.nasa_moc import set_theta_b

    Rt = 0.020
    epsilon = 10.0
    length_pct = 80.0
    # Same exit-station convention as solve_rao_bvp (_target_length).
    from raosim.rao_variational import _target_length
    Ln = _target_length(Rt, epsilon, length_pct)
    Re = math.sqrt(epsilon) * Rt

    topo, kern = set_theta_b(
        Rt, epsilon, length_pct, GAMMA, 0.01,
        theta_b_init_deg=21.87, n_kernel=24, n_de_points=24,
        starting_line_method="kliegel_levine", L_target=Ln,
        Ru=1.5 * Rt, end_condition="fixed_end", max_iter=30,
    )
    assert kern.reached_wall
    # theta_B converged above the historic cap, in the expected band.
    assert 24.0 < math.degrees(topo.theta_B) < 27.0
    # Exact closure at the commanded exit station.
    assert abs(topo.E.x - Ln) / Ln < 1e-3
    assert abs(topo.E.r - Re) / Re < 1e-3
    # Mass budget: DE carries exactly the wall-to-D mass.
    assert topo.mass_DE == pytest.approx(topo.mass_BD, rel=1e-6)
    # D interior on BD (not collapsed onto B or the axis).
    assert 0.02 < topo.d_fraction < 0.95


@pytest.mark.slow
def test_fixed_end_topology_wall_is_bell_shaped():
    """Kernel arc wall + BDE wall from the fixed-end topology = TOP bell.

    This is the geometry-level counterpart of the solved-CE convergence
    research cases in test_jax_convergence.py: slope must peak near theta_B
    right after the throat arc and decrease monotonically — no mid-bell
    flare (the old Rao-case contour peaked 35.6 deg at 60% length).
    """
    from raosim.nasa_moc import set_theta_b
    from raosim.rao_variational import _target_length

    Rt = 0.020
    Ln = _target_length(Rt, 10.0, 80.0)
    Re = math.sqrt(10.0) * Rt
    topo, kern = set_theta_b(
        Rt, 10.0, 80.0, GAMMA, 0.01,
        theta_b_init_deg=21.87, n_kernel=48, n_de_points=40,
        starting_line_method="kliegel_levine", L_target=Ln,
        Ru=1.5 * Rt, end_condition="fixed_end", max_iter=30,
    )
    bfe = calc_bde_region(kern, topo)
    assert bfe.complete_remaining_mesh
    assert bfe.wall_contour_complete

    kernel_wall = [(row[0].x, row[0].r) for row in kern.rrcs if row]
    bfe_wall = [(p.x, p.r) for p in bfe.wall_contour]
    wall = np.asarray(kernel_wall + bfe_wall)

    # Exit lands on the commanded station.
    assert wall[-1, 0] == pytest.approx(Ln, rel=1e-3)
    assert wall[-1, 1] == pytest.approx(Re, rel=1e-3)

    ang = np.degrees(np.arctan2(np.diff(wall[:, 1]), np.diff(wall[:, 0])))
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(wall[:, 0]),
                                                  np.diff(wall[:, 1])))])
    i_pk = int(np.argmax(ang))
    theta_b_deg = math.degrees(topo.theta_B)
    # Peak sits just after the throat arc, near theta_B (small overshoot
    # from arc/BFE attachment discretisation is tolerated).
    assert s[i_pk] / s[-1] < 0.10
    assert ang.max() == pytest.approx(theta_b_deg, abs=1.5)
    # Monotone decreasing after the peak; physical exit angle.
    assert np.all(np.diff(ang[i_pk:]) <= 0.25)
    assert 6.0 <= ang[-1] <= 14.0
