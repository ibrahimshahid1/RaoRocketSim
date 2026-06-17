"""Generate a 3-D regen-cooled nozzle (wall + cooling channels).

Builds the nozzle wall as a surface of revolution and the N
regenerative cooling channels wrapping it (axial, or helical "coils"),
exports a binary STL for CAD/print and a 3-D PNG.  Optionally colours
the channels by the gas-side wall temperature from the coupled cooling
analysis (the throat hot spot painted onto the 3-D model).

Examples
--------
    # 80 axial channels, copper, coloured by wall temperature
    PYTHONPATH=. python scripts/generate_regen_nozzle.py --channels 80 --thermal

    # 24 helical coils, 3 turns
    PYTHONPATH=. python scripts/generate_regen_nozzle.py \
        --channels 24 --channel-width 0.0018 --helix-turns 3 --out builds/coil
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")

from raosim.design import CoolingSpec, MaterialSpec
from raosim.nozzle_geometry import bell_nozzle_contour
from raosim.physics import bartz_heat_flux, regenerative_cooling_analysis
from raosim.propellants import custom_propellant
from raosim.regen_geometry import generate_regen_nozzle


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rt", type=float, default=0.020, help="throat radius [m]")
    ap.add_argument("--epsilon", type=float, default=10.0)
    ap.add_argument("--length-pct", type=float, default=80.0)
    ap.add_argument("--gamma", type=float, default=1.24)
    ap.add_argument("--channels", type=int, default=80)
    ap.add_argument("--channel-width", type=float, default=0.0008, help="[m]")
    ap.add_argument("--channel-height", type=float, default=0.0025, help="[m]")
    ap.add_argument("--wall-thickness", type=float, default=0.001, help="[m]")
    ap.add_argument("--helix-turns", type=float, default=0.0,
                    help="0 = axial channels; >0 = helical coils")
    ap.add_argument("--pc", type=float, default=7.0e6, help="chamber pressure [Pa]")
    ap.add_argument("--thermal", action="store_true",
                    help="colour channels by the coupled-cooling wall temperature")
    ap.add_argument("--out", type=Path, default=Path("builds/regen_nozzle"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    contour = bell_nozzle_contour(
        Rt=args.rt, epsilon=args.epsilon, gamma=args.gamma,
        length_pct=args.length_pct)
    cooling = CoolingSpec(
        method="regenerative", coolant="rp1",
        channel_count=args.channels, channel_width=args.channel_width,
        channel_height=args.channel_height, coolant_mass_flow=10.0,
        coolant_inlet_temperature=300.0, max_wall_temperature=1200.0)

    cooling_result = None
    if args.thermal:
        prop = custom_propellant(gamma=args.gamma, Mw=0.022, Tc=3500.0)
        hf = bartz_heat_flux(contour, args.pc, prop, wall_temperature=900.0)
        cooling_result = regenerative_cooling_analysis(
            hf, contour, cooling, MaterialSpec(conductivity=350.0),
            args.wall_thickness, prop, args.pc)
        print(f"peak wall T = {cooling_result['peak_gas_side_wall_temperature']:.0f} K, "
              f"cooling margin = {cooling_result['cooling_margin']:.2f}, "
              f"Δp = {cooling_result['coolant_pressure_drop']/1e5:.1f} bar")

    res = generate_regen_nozzle(
        contour, cooling, args.wall_thickness,
        helix_turns=args.helix_turns,
        stl_path=f"{args.out}.stl", png_path=f"{args.out}.png",
        cooling_result=cooling_result)
    s = res["summary"]
    print(f"channels: {s['n_channels']} ({'helix %.1f turns' % s['helix_turns'] if s['helix_turns'] else 'axial'}), "
          f"fit={s['channels_fit']}")
    print(f"wrote {res['stl_path']} ({res['n_triangles']} triangles) and {res['png_path']}")
    if not s["channels_fit"]:
        print("WARNING: channels exceed the throat circumference; reduce count/width.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
