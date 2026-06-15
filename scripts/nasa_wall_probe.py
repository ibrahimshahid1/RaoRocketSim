"""Phase 12.5 probe: BDE wall constructor vs NASA wall.out (M3.5Perf).

Perfect-nozzle branch: D = axis end of BD, mdot_BD = full kernel mass,
DE = uniform-state characteristic from D (C++ FindPointE PERFECT-AXI
closed form).  Then calc_bde_region builds the wall; compare against
the i=0 trace of outputs_M3.5Perf/wall.out.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from raosim.nasa_moc import (MOCNode, RaoTopology, build_kernel,
                             calc_bde_region, calc_massflow_along_rrc)
from raosim.legacy_io import parse_wall_out

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Three-Dimensional-Nozzle-Design-Code-master" / "MOC_Grid_BDE" / "outputs_M3.5Perf"
G = 1.4

kernel = build_kernel(1.0, 1.0, math.radians(15.2196), G, 101,
                      starting_line_method="nasa_visible_kliegel_levine")
bd = kernel.bd                       # wall-first
print(f"kernel rows={len(kernel.rrcs)} bd_n={len(bd)} "
      f"axis end: x={bd[-1].x:.5f} r={bd[-1].r:.6f} M={bd[-1].M:.4f}")

# Perfect: D = axis end; mdot = full BD mass (our normalised convention)
massflow = calc_massflow_along_rrc(bd, G)
mdot = float(massflow[0])
D = bd[-1]
rho, u = D.rho, D.u
rE = math.sqrt(mdot / (math.pi * rho * u))
muD = D.mu
xE = D.x + rE / math.tan(muD)        # dr/dx = tan(theta+mu) = tan(mu), theta=0
print(f"mdot={mdot:.6f}  rE={rE:.5f} (NASA exit R/R*~{math.sqrt(3.5**0 + 0)+0:.0f}...)  xE={xE:.5f}")

n_de = 40
de = [MOCNode(D.x + (xE - D.x) * t, rE * t, max(D.M, 1.000001), 0.0, G)
      for t in np.linspace(0.0, 1.0, n_de)]
topo = RaoTopology(
    B=bd[0].to_flow_node(), BD=tuple(p.to_flow_node() for p in bd),
    D=D.to_flow_node(), DE=tuple(p.to_flow_node() for p in de),
    E=de[-1].to_flow_node(), d_fraction=1.0,
    mass_BD=mdot, mass_DE=mdot, thrust_coefficient=float("nan"),
    theta_control=0.0, theta_B=float(kernel.theta_B),
    rao_stationarity_residual=float("nan"),
)
bfe = calc_bde_region(kernel, topo)
print(f"bfe rows={len(bfe.grid_rows)} wall pts={len(bfe.wall_contour)} "
      f"mesh={bfe.complete_remaining_mesh} wall_complete={bfe.wall_contour_complete}")

kernel_wall = [(r[0].x, r[0].r, math.degrees(r[0].theta)) for r in kernel.rrcs if r]
bfe_wall = [(p.x, p.r, math.degrees(p.theta)) for p in bfe.wall_contour]
wall = np.asarray(kernel_wall + bfe_wall)

nasa = parse_wall_out(OUT / "wall.out")
cols = {k.lower(): k for k in nasa.columns}
nx = nasa.column([c for c in nasa.columns if "x" in c.lower()][0])
nr = nasa.column([c for c in nasa.columns if "r/" in c.lower() or c.lower().startswith("r")][0])
print(f"nasa wall pts={len(nx)} exit=({nx[-1]:.4f},{nr[-1]:.4f})   "
      f"py exit=({wall[-1,0]:.4f},{wall[-1,1]:.4f})")

xc = np.linspace(max(nx.min(), wall[:, 0].min()),
                 min(nx.max(), wall[:, 0].max()), 200)
ri = np.interp(xc, wall[:, 0], wall[:, 1])
ni = np.interp(xc, nx, nr)
rms = float(np.sqrt(np.mean((ri - ni) ** 2)))
print(f"wall r RMS vs NASA = {rms:.4e}  max = {np.max(np.abs(ri-ni)):.4e}")
sl = np.degrees(np.arctan2(np.diff(wall[:, 1]), np.diff(wall[:, 0])))
print("py wall slope every 12th:", np.round(sl[::12], 1))
