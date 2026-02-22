#!/usr/bin/env python3
"""
main.py – Interactive CLI for the Rao Bell Nozzle Design Toolbox.

Usage:
    python main.py              # interactive mode
    python main.py --help       # show help
"""

from __future__ import annotations
import argparse
import math
import sys
from pathlib import Path

from raosim.propellants import (
    get_propellant, custom_propellant, list_propellants, Propellant,
)
from raosim.gas_dynamics import (
    mach_from_area_ratio, expansion_ratio_from_pressure,
    isentropic_pressure_ratio, prandtl_meyer,
)
from raosim.nozzle_geometry import bell_nozzle_contour, lookup_angles
from raosim.engine import compute_engine_performance, g0
from raosim.export import export_csv, export_stl
from raosim.plotting import plot_nozzle_2d, plot_nozzle_3d, plot_curvature
from raosim.atmosphere import pressure as atm_pressure


# ── Pretty-printing helpers ──────────────────────────────────────────

def _header():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║           Rao Bell Nozzle Design Toolbox  v1.0         ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


def _ask(prompt: str, default=None, cast=float):
    """Prompt user; return default if blank."""
    suffix = f" [{default}]" if default is not None else ""
    raw = input(f"  {prompt}{suffix}: ").strip()
    if raw == "":
        if default is None:
            raise ValueError("No default; value is required.")
        return cast(default) if cast else default
    return cast(raw)


def _ask_str(prompt: str, default: str = "") -> str:
    return _ask(prompt, default=default, cast=str)


# ── Main interactive flow ────────────────────────────────────────────

def main():
    _header()

    # ─────────── 1. Propellant selection ─────────────────────────
    avail = list_propellants()
    print("  Available propellants:")
    for i, name in enumerate(avail, 1):
        print(f"    {i}. {name}")
    print(f"    {len(avail)+1}. Custom (enter γ, Mw, Tc manually)")
    print()

    choice = _ask("Select propellant number", default=1, cast=int)
    if choice <= len(avail):
        prop = get_propellant(avail[choice - 1])
    else:
        gamma = _ask("γ (ratio of specific heats)", default=1.24)
        Mw = _ask("Mw (mean molecular weight, kg/mol)", default=0.022)
        Tc = _ask("Tc (chamber temperature, K)", default=3500)
        eta = _ask("η_Isp (efficiency factor, 0-1)", default=0.95)
        prop = custom_propellant(gamma, Mw, Tc, eta)

    print(f"\n  ✓ Propellant: {prop.name}")
    print(f"    γ = {prop.gamma},  Mw = {prop.Mw*1000:.1f} g/mol,  "
          f"Tc = {prop.Tc:.0f} K")
    print(f"    R_gas = {prop.R_gas:.2f} J/(kg·K),  "
          f"c* = {prop.c_star:.1f} m/s")
    print()

    # ─────────── 2. Chamber & nozzle parameters ─────────────────
    Pc_bar = _ask("Chamber pressure Pc [bar]", default=45)
    Pc = Pc_bar * 1e5

    Pa_kPa = _ask("Ambient pressure Pa [kPa] (101.325 = sea level)", default=101.325)
    Pa = Pa_kPa * 1e3

    Rt_mm = _ask("Throat radius Rt [mm]", default=20.0)
    Rt = Rt_mm / 1000.0

    # ─────────── 3. Expansion ratio or compute from pressures ────
    print()
    print("  How to set the expansion ratio?")
    print("    1. Compute from Pc/Pa  (matched expansion at ambient)")
    print("    2. Specify ε directly")
    print("    3. Specify exit radius Re directly")
    eps_mode = _ask("Choice", default=1, cast=int)

    if eps_mode == 1:
        epsilon, Me = expansion_ratio_from_pressure(Pc, Pa, prop.gamma)
    elif eps_mode == 2:
        epsilon = _ask("Expansion ratio ε = Ae/At", default=10.0)
        Me = mach_from_area_ratio(epsilon, prop.gamma)
    else:
        Re_mm = _ask("Exit radius Re [mm]", default=60.0)
        Re = Re_mm / 1000.0
        epsilon = (Re / Rt) ** 2
        Me = mach_from_area_ratio(epsilon, prop.gamma)

    Re = math.sqrt(epsilon) * Rt
    print(f"\n  ✓ ε = {epsilon:.2f}   (Me = {Me:.4f})")
    print(f"    Re = {Re*1000:.2f} mm")

    # ─────────── 4. Bell length fraction & angles ────────────────
    print()
    length_pct = _ask("Bell length [% of 15° cone] (60–100)",
                      default=80.0)

    tn_default, te_default = lookup_angles(epsilon, length_pct)
    print(f"\n  Rao/NASA table → θ_n = {tn_default:.1f}°, "
          f"θ_e = {te_default:.1f}°")
    override = _ask_str("Use these angles? [Y/n]", default="y").lower()

    if override.startswith("n"):
        theta_n = _ask("θ_n (initial wall angle) [°]", default=tn_default)
        theta_e = _ask("θ_e (exit wall angle) [°]", default=te_default)
    else:
        theta_n = tn_default
        theta_e = te_default

    # ─────────── 5. Generate contour ─────────────────────────────
    contour = bell_nozzle_contour(
        Rt=Rt, epsilon=epsilon,
        theta_n_deg=theta_n, theta_e_deg=theta_e,
        length_pct=length_pct,
    )

    print(f"\n  ✓ Contour generated: {len(contour['x'])} points")
    print(f"    Bell length Ln = {contour['Ln']*1000:.2f} mm")
    print(f"    Upstream fillet Ru = {contour['Ru']*1000:.2f} mm")
    print(f"    Downstream fillet Rd = {contour['Rd']*1000:.2f} mm")

    # ─────────── 6. Engine performance ───────────────────────────
    perf = compute_engine_performance(Pc, Pa, Rt, epsilon, prop)

    print()
    print("  ── Engine Performance ──────────────────────────────────")
    print(f"    Thrust (F)        = {perf.thrust:.2f} N  "
          f"({perf.thrust/1000:.2f} kN)")
    print(f"    Mass flow (ṁ)     = {perf.m_dot:.4f} kg/s")
    print(f"    Isp               = {perf.Isp:.1f} s")
    print(f"    Ve                = {perf.Ve:.1f} m/s")
    print(f"    Cf (ideal)        = {perf.Cf_ideal:.4f}")
    print(f"    Cf (actual, η={perf.eta_Isp}) = {perf.Cf_actual:.4f}")
    print(f"    c*                = {perf.c_star:.1f} m/s")
    print(f"    Exit Mach         = {perf.Me:.4f}")
    print(f"    Exit pressure     = {perf.Pe:.0f} Pa  "
          f"({perf.Pe/1000:.2f} kPa)")
    print(f"    Pe/Pc             = {perf.Pe_over_Pc:.6f}")

    # Over/underexpansion check
    if perf.Pe < Pa:
        print(f"    ⚠  Overexpanded (Pe < Pa by {(Pa-perf.Pe)/1000:.2f} kPa)")
    elif perf.Pe > Pa * 1.05:
        print(f"    ⚠  Underexpanded (Pe > Pa by {(perf.Pe-Pa)/1000:.2f} kPa)")
    else:
        print(f"    ✓  Near-matched expansion at this ambient pressure")

    # ─────────── 7. Export ───────────────────────────────────────
    print()
    n_csv = int(_ask("Number of CSV points", default=301))
    csv_name = _ask_str("CSV file name", default="rao_nozzle_profile.csv")
    csv_path = export_csv(contour['x'], contour['y'], csv_name, n_csv)
    print(f"  → CSV written: {csv_path}")

    do_stl = _ask_str("Export STL for CAD import? [Y/n]", default="y").lower()
    if not do_stl.startswith("n"):
        stl_name = _ask_str("STL file name", default="rao_nozzle.stl")
        n_ang = int(_ask("Angular resolution (faces)", default=64))
        stl_path = export_stl(contour['x'], contour['y'], stl_name, n_ang)
        print(f"  → STL written: {stl_path}")

    # ─────────── 8. Plot ─────────────────────────────────────────
    do_plot = _ask_str("Show plots? [Y/n]", default="y").lower()
    if not do_plot.startswith("n"):
        plot_nozzle_2d(contour, show=True)
        plot_nozzle_3d(contour, show=True)
        do_curv = _ask_str("Show curvature distribution? [Y/n]",
                           default="y").lower()
        if not do_curv.startswith("n"):
            plot_curvature(contour, show=True)

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
