#!/usr/bin/env python3
"""Offline CEA property-table sampler (host-only; needs RocketCEA).

Phase-2 workflow (docs/DIFFERENTIABLE_ENGINE_MDO_PLAN.md §12.2 row 2):
sample frozen chamber properties on a (Pc, O/F) grid, save as .npz, load in
the differentiable layer via ``raosim.mdo.properties.load_chamber_surfaces``.

Example:
    python scripts/sample_cea_surface.py --oxidizer LOX --fuel RP-1 \
        --pc-min 1.5e6 --pc-max 6.0e6 --n-pc 13 \
        --of-min 1.6 --of-max 3.2 --n-of 13 \
        --out builds/cea_tables/lox_rp1_frozen.npz

Validation: hold out every 3rd grid point with --holdout and the script
reports the surface-vs-CEA error on the held-out points (the Phase-2 gate).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oxidizer", required=True)
    ap.add_argument("--fuel", required=True)
    ap.add_argument("--pc-min", type=float, required=True)
    ap.add_argument("--pc-max", type=float, required=True)
    ap.add_argument("--n-pc", type=int, default=13)
    ap.add_argument("--of-min", type=float, required=True)
    ap.add_argument("--of-max", type=float, required=True)
    ap.add_argument("--n-of", type=int, default=13)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout", action="store_true",
                    help="also sample midpoints and report interpolation error")
    args = ap.parse_args()

    from raosim.mdo.properties import (
        load_chamber_surfaces, sample_cea_tables, save_tables,
    )

    Pc_grid = np.linspace(args.pc_min, args.pc_max, args.n_pc)
    OF_grid = np.linspace(args.of_min, args.of_max, args.n_of)
    tables = sample_cea_tables(Pc_grid, OF_grid,
                               oxidizer=args.oxidizer, fuel=args.fuel)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_tables(str(out), tables, oxidizer=args.oxidizer, fuel=args.fuel)
    print(f"wrote {out}  grid {args.n_pc}x{args.n_of}")

    if args.holdout:
        surf = load_chamber_surfaces(str(out))
        Pc_mid = 0.5 * (Pc_grid[:-1] + Pc_grid[1:])
        OF_mid = 0.5 * (OF_grid[:-1] + OF_grid[1:])
        ref = sample_cea_tables(Pc_mid, OF_mid,
                                oxidizer=args.oxidizer, fuel=args.fuel)
        for key in ("gamma", "Tc", "R_gas"):
            grid_vals = np.array([
                [float(getattr(surf, key)(pc, of)) for of in OF_mid]
                for pc in Pc_mid
            ])
            rel = np.abs(grid_vals - ref[key]) / np.abs(ref[key])
            print(f"holdout {key}: max rel err {rel.max():.3e} "
                  f"(Phase-2 gate: report + compare against tolerance)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
