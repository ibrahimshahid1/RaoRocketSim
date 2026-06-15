"""Render the Phase-12.6 reference topology (fixed-end closure) to PNG.

Builds the full-form RaoTopology for the reference design point
(epsilon=10, L80, Rd=0.382Rt, Ru=1.5Rt) and writes:

    builds/reference_topology.png    plot_topology overlay
    builds/reference_wall.csv        full wall (throat arc + streamline_BE)

This is the geometrically-sane characteristic closure (NOT the Rao
optimum — see moc_topology.build_reference_topology's docstring).

Run:  PYTHONPATH=. MPLBACKEND=Agg python scripts/plot_reference_topology.py
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np

from raosim.moc_topology import build_reference_topology
from raosim.plotting import plot_topology

OUT = Path("builds")


def main() -> None:
    topo = build_reference_topology(0.020, 10.0, 80.0, 1.4, 0.01,
                                    n_kernel=48, n_de_points=40)
    rep = topo.closure_report()
    print(f"theta_B = {math.degrees(topo.theta_B):.4f} deg   "
          f"kdf = {topo.d_fraction:.4f}")
    for k, v in rep.items():
        print(f"  {k:22s} {v:.3e}")

    wall = topo.full_wall()
    ang = np.degrees(np.arctan2(np.diff(wall[:, 1]), np.diff(wall[:, 0])))
    print(f"wall: n={len(wall)} exit=({wall[-1, 0]:.6f},{wall[-1, 1]:.6f}) "
          f"peak={ang.max():.2f} deg exit={ang[-1]:.2f} deg")

    OUT.mkdir(parents=True, exist_ok=True)
    fig = plot_topology(topo, save_path=str(OUT / "reference_topology.png"))
    fig.clf()
    np.savetxt(OUT / "reference_wall.csv", wall, delimiter=",",
               header="x_m,r_m", comments="")
    print(f"wrote {OUT / 'reference_topology.png'} and "
          f"{OUT / 'reference_wall.csv'}")


if __name__ == "__main__":
    main()
