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
