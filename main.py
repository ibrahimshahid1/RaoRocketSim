#!/usr/bin/env python3
"""
main.py – CLI for the Rao Bell Nozzle Design Toolbox.

Usage:
    python main.py                        # interactive mode
    python main.py --help                 # show all flags
    python main.py --Rt 25 --Pc 60 \\
        --propellant LOX/LCH4 --epsilon 12 \\
        --output nozzle.csv               # batch mode (no prompts)
    python main.py --sweep epsilon 4 50 20  # parameter sweep
"""

from __future__ import annotations
import argparse
import json
import math
import sys
import numpy as np
from pathlib import Path

from raosim.cea import propellant_from_request
from raosim.design import throat_radius_for_target_thrust
from raosim.propellants import (
    get_propellant, custom_propellant, list_propellants, Propellant,
)
from raosim.gas_dynamics import (
    mach_from_area_ratio, expansion_ratio_from_pressure,
    isentropic_pressure_ratio, prandtl_meyer,
)
from raosim.nozzle_geometry import bell_nozzle_contour, lookup_angles
from raosim.engine import compute_engine_performance, g0
from raosim.export import export_csv, export_stl, export_step, package_ipt_request
from raosim.plotting import plot_nozzle_2d, plot_nozzle_3d, plot_curvature
from raosim.atmosphere import pressure as atm_pressure
from raosim.wall_pressure import wall_pressure_distribution, plot_wall_pressure
from raosim.separation import check_separation, separation_summary
from raosim.trade_study import (
    sweep_epsilon, sweep_Pc, sweep_Rt, plot_trade_study,
)
from raosim.altitude_performance import (
    altitude_performance_map, plot_altitude_performance,
)
from raosim.chamber_geometry import chamber_contour, full_engine_contour
from raosim.build_log import create_build_dir, write_metadata
from raosim.validation import evaluate_design_gates


def _header():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         Rao Bell Nozzle Design Toolbox  v2.0           ║")
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rao Bell Nozzle Design Toolbox v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  Interactive:   python main.py
  Batch:         python main.py --propellant LOX/RP-1 --Pc 45 --Rt 20 --epsilon 10
  Sweep:         python main.py --propellant LOX/LCH4 --Pc 60 --Rt 25 \\
                     --sweep epsilon 4 50 20
""",
    )

    p.add_argument('--propellant', type=str, default=None,
                   help='Propellant name (e.g. LOX/RP-1, LOX/LCH4)')
    p.add_argument('--oxidizer', type=str, default=None,
                   help='CEA oxidizer name (e.g. LOX)')
    p.add_argument('--fuel', type=str, default=None,
                   help='CEA fuel name (e.g. RP-1)')
    p.add_argument('--of', type=float, default=None,
                   help='Mixture ratio O/F for CEA-backed properties')
    p.add_argument('--cea', action='store_true',
                   help='Use RocketCEA/NASA CEA thermochemistry when available')
    p.add_argument('--Pc', type=float, default=None,
                   help='Chamber pressure [bar]')
    p.add_argument('--Pa', type=float, default=None,
                   help='Ambient pressure [kPa] (default 101.325)')
    p.add_argument('--Rt', type=float, default=None,
                   help='Throat radius [mm]')
    p.add_argument('--target-thrust', type=float, default=None,
                   help='Target thrust [N]; used to size Rt if --Rt is omitted')
    p.add_argument('--epsilon', type=float, default=None,
                   help='Expansion ratio Ae/At')
    p.add_argument('--length-pct', type=float, default=80.0,
                   help='Bell length %% of 15° cone (default 80)')
    p.add_argument('--method', type=str, default='bezier',
                    choices=['bezier', 'moc', 'rao'],
                    help='Contour method: bezier (default), moc (direct wall optimization), '
                         'or rao (calculus-of-variations control-surface optimization)')
    p.add_argument('--compare', action='store_true',
                    help='Compare conical, Bézier, and Rao contours side by side')
    p.add_argument('--theta-n', type=float, default=None,
                   help='Override initial wall angle θ_n [°]')
    p.add_argument('--theta-e', type=float, default=None,
                   help='Override exit wall angle θ_e [°]')


    p.add_argument('--gamma', type=float, default=None,
                   help='Custom γ (requires --Mw and --Tc)')
    p.add_argument('--Mw', type=float, default=None,
                   help='Custom molecular weight [kg/mol]')
    p.add_argument('--Tc', type=float, default=None,
                   help='Custom chamber temperature [K]')
    p.add_argument('--eta', type=float, default=0.95,
                   help='Isp efficiency factor (default 0.95)')


    p.add_argument('--output', '--csv', type=str, default=None,
                   help='CSV output path')
    p.add_argument('--stl', type=str, default=None,
                   help='STL output path')
    p.add_argument('--cad', type=str, default='none',
                   choices=['none', 'step', 'ipt', 'both'],
                   help='Solid CAD export: STEP, IPT manifest, or both')
    p.add_argument('--wall-thickness', type=float, default=None,
                   help='Solid nozzle wall thickness [mm] for STEP/STL CAD')
    p.add_argument('--flange-od', type=float, default=None,
                   help='Optional inlet flange outer diameter [mm]')
    p.add_argument('--flange-length', type=float, default=None,
                   help='Optional inlet flange axial length [mm]')
    p.add_argument('--design-report', action='store_true',
                   help='Write design-gate report JSON')
    p.add_argument('--strict-gates', action='store_true',
                   help='Exit nonzero if design gates fail')
    p.add_argument('--n-csv', type=int, default=301,
                   help='Number of CSV points (default 301)')
    p.add_argument('--n-angular', type=int, default=64,
                   help='STL angular resolution (default 64)')
    p.add_argument('--no-plot', action='store_true',
                   help='Suppress all plots')


    p.add_argument('--wall-pressure', action='store_true',
                   help='Compute and plot wall pressure distribution')
    p.add_argument('--separation', action='store_true',
                   help='Run separation check')
    p.add_argument('--sep-method', type=str, default='schmucker',
                   choices=['summerfield', 'kalt_badal', 'schmucker'],
                   help='Separation criterion (default schmucker)')
    p.add_argument('--altitude-map', action='store_true',
                   help='Compute and plot altitude performance map')
    p.add_argument('--chamber', action='store_true',
                   help='Generate combustion chamber geometry')
    p.add_argument('--L-star', type=float, default=1.0,
                   help='Chamber characteristic length L* [m] (default 1.0)')
    p.add_argument('--contraction-ratio', type=float, default=2.5,
                   help='Chamber contraction ratio Ac/At (default 2.5)')


    p.add_argument('--sweep', nargs=4, metavar=('VAR', 'MIN', 'MAX', 'N'),
                   help='Sweep a variable: --sweep epsilon 4 50 20')

    return p


def is_batch(args) -> bool:
    """Return True if enough args are given to skip interactive prompts."""
    has_prop = args.propellant is not None or args.gamma is not None or args.cea
    has_size = args.Rt is not None or args.target_thrust is not None
    return (has_prop and args.Pc is not None and
            has_size and args.epsilon is not None)


def run_batch(args):
    """Non-interactive mode: all params from argparse."""
    _header()


    warnings: list[str] = []

    if args.gamma is not None:
        Mw = args.Mw or 0.022
        Tc = args.Tc or 3500
        prop = custom_propellant(args.gamma, Mw, Tc, args.eta)
    elif args.cea:
        prop, prop_warnings = propellant_from_request(
            propellant_name=args.propellant,
            use_cea=True,
            Pc=args.Pc * 1e5,
            mixture_ratio=args.of,
            oxidizer=args.oxidizer,
            fuel=args.fuel,
            eta_Isp=args.eta,
        )
        warnings.extend(prop_warnings)
    else:
        prop = get_propellant(args.propellant)

    Pc = args.Pc * 1e5
    Pa = (args.Pa if args.Pa is not None else 101.325) * 1e3
    epsilon = args.epsilon
    length_pct = args.length_pct

    if args.Rt is not None:
        Rt = args.Rt / 1000.0
    elif args.target_thrust is not None:
        Rt = throat_radius_for_target_thrust(
            args.target_thrust, Pc, Pa, epsilon, prop,
        )
        warnings.append(
            f"Sized Rt from target thrust: Rt = {Rt * 1000:.3f} mm."
        )
    else:
        raise ValueError("Either --Rt or --target-thrust is required.")

    wall_thickness = (
        args.wall_thickness / 1000.0 if args.wall_thickness is not None else None
    )
    flange_od = args.flange_od / 1000.0 if args.flange_od is not None else None
    flange_length = (
        args.flange_length / 1000.0 if args.flange_length is not None else None
    )
    if args.cad in {'step', 'ipt', 'both'} and wall_thickness is None:
        raise ValueError("--cad STEP/IPT export requires --wall-thickness [mm].")
    if (flange_od is None) != (flange_length is None):
        raise ValueError("--flange-od and --flange-length must be supplied together.")

    if args.method == 'moc':
        contour = bell_nozzle_contour(Rt, epsilon, method='moc',
                                       length_pct=length_pct,
                                       gamma=prop.gamma)
    elif args.method == 'rao':
        contour = bell_nozzle_contour(Rt, epsilon, method='rao',
                                       length_pct=length_pct,
                                       gamma=prop.gamma)
    else:
        theta_n = args.theta_n
        theta_e = args.theta_e
        if theta_n is None or theta_e is None:
            tn_l, te_l = lookup_angles(epsilon, length_pct)
            theta_n = theta_n or tn_l
            theta_e = theta_e or te_l
        contour = bell_nozzle_contour(Rt, epsilon, theta_n, theta_e, length_pct,
                                       gamma=prop.gamma)
    perf = compute_engine_performance(Pc, Pa, Rt, epsilon, prop)
    gate_report = evaluate_design_gates(
        contour, Pc, Pa, prop.gamma,
        wall_thickness=wall_thickness,
        flange_od=flange_od,
        flange_length=flange_length,
    )
    warnings.extend(contour.get('warnings', []))
    warnings.extend(gate_report.warnings)
    warnings = _dedupe(warnings)

    _print_summary(prop, contour, perf, Pc, Pa, Rt, epsilon)
    _print_warnings(warnings)


    if args.separation:
        sep = check_separation(contour, Pc, Pa, prop.gamma, args.sep_method)
        print(separation_summary(sep))


    if args.wall_pressure:
        wp = wall_pressure_distribution(contour, Pc, prop.gamma)
        if wp['monotonic']:
            print("  ✓ Wall pressure is monotonically decreasing (no sep risk)")
        else:
            print(f"  ⚠ Wall pressure non-monotonic at {len(wp['violation_indices'])} points")
        if not args.no_plot:
            plot_wall_pressure(wp)


    if args.chamber:
        ch = chamber_contour(Rt, args.L_star, args.contraction_ratio)
        engine = full_engine_contour(ch, contour)
        print(f"\n  ── Chamber Geometry ────────────────────────────────────")
        print(f"    Chamber radius Rc = {ch['Rc']*1000:.2f} mm")
        print(f"    Cylinder length   = {ch['Lc']*1000:.1f} mm")
        print(f"    Convergent length = {ch['L_conv']*1000:.1f} mm")
        print(f"    Chamber volume    = {ch['V_chamber']*1e6:.2f} cm³")
        print(f"    L* = {ch['L_star']:.3f} m")


    if args.altitude_map and not args.no_plot:
        apm = altitude_performance_map(Pc, Rt, epsilon, prop, contour)
        if apm['h_sep_onset'] is not None:
            print(f"\n  Separation clears at {apm['h_sep_onset']/1000:.1f} km altitude")
        plot_altitude_performance(apm)


    build_dir, version = create_build_dir()
    output_files: list[str] = []


    csv_name = args.output or "rao_nozzle_profile.csv"
    csv_path = export_csv(contour['x'], contour['y'],
                          build_dir / Path(csv_name).name, args.n_csv)
    output_files.append(csv_path.name)
    print(f"  → CSV: {csv_path}")


    if args.stl:
        stl_path = export_stl(contour['x'], contour['y'],
                              build_dir / Path(args.stl).name, args.n_angular,
                              wall_thickness=wall_thickness,
                              flange_od=flange_od,
                              flange_length=flange_length)
        output_files.append(stl_path.name)
        print(f"  → STL: {stl_path}")

    step_path = None
    if args.cad in {'step', 'ipt', 'both'}:
        step_name = "rao_nozzle.step"
        step_path = export_step(
            contour['x'], contour['y'], build_dir / step_name, args.n_angular,
            wall_thickness=wall_thickness,
            flange_od=flange_od,
            flange_length=flange_length,
            metadata={
                "design_status": contour.get("design_status"),
                "hardware_qualified": False,
                "gate_passed": gate_report.passed,
            },
        )
        output_files.append(step_path.name)
        print(f"  → STEP: {step_path}")

    if args.cad in {'ipt', 'both'} and step_path is not None:
        ipt_manifest = package_ipt_request(
            step_path, build_dir / "rao_nozzle_ipt_manifest.json",
            metadata={
                "design_status": contour.get("design_status"),
                "hardware_qualified": False,
                "gate_passed": gate_report.passed,
            },
        )
        output_files.append(ipt_manifest.name)
        print(f"  → IPT manifest: {ipt_manifest}")

    if args.design_report:
        report_path = build_dir / "design_report.json"
        report_path.write_text(
            json.dumps(gate_report.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        output_files.append(report_path.name)
        print(f"  → Design report: {report_path}")


    params = {
        "Propellant": prop.name,
        "Pc [bar]": f"{Pc / 1e5:.2f}",
        "Pa [kPa]": f"{Pa / 1e3:.3f}",
        "Rt [mm]": f"{Rt * 1000:.2f}",
        "Target thrust [N]": (
            f"{args.target_thrust:.2f}" if args.target_thrust is not None else "N/A"
        ),
        "Epsilon (Ae/At)": f"{epsilon:.2f}",
        "Bell length %": f"{length_pct:.1f}",
        "Theta_n [deg]": f"{contour['theta_n']:.2f}",
        "Theta_e [deg]": f"{contour['theta_e']:.2f}",
        "Design status": contour.get("design_status", "unknown"),
        "Hardware qualified": "False",
        "Qualification note": contour.get("qualification_note", ""),
        "Gamma": f"{prop.gamma}",
        "Mw [kg/mol]": f"{prop.Mw}",
        "Tc [K]": f"{prop.Tc:.0f}",
        "Eta_Isp": f"{prop.eta_Isp}",
    }
    perf_dict = {
        "Thrust [N]": f"{perf.thrust:.2f}",
        "Thrust [kN]": f"{perf.thrust / 1000:.3f}",
        "Isp [s]": f"{perf.Isp:.1f}",
        "Mass flow [kg/s]": f"{perf.m_dot:.4f}",
        "Ve [m/s]": f"{perf.Ve:.1f}",
        "Exit Mach": f"{perf.Me:.4f}",
        "Exit pressure [Pa]": f"{perf.Pe:.0f}",
        "Cf ideal": f"{perf.Cf_ideal:.4f}",
        "Cf actual": f"{perf.Cf_actual:.4f}",
        "c* [m/s]": f"{perf.c_star:.1f}",
    }
    meta_path = write_metadata(build_dir, version=version, mode="batch",
                               params=params, performance=perf_dict,
                               warnings=warnings,
                               gate_report=gate_report.to_dict(),
                               files=output_files)
    output_files.append(meta_path.name)

    print(f"\n  📁 Build v{version:03d}: {build_dir}")
    if args.strict_gates and not gate_report.passed:
        print("  ✗ Strict gates enabled and one or more design gates failed.")
        sys.exit(2)


    # ── Contour comparison mode ──
    if args.compare:
        from raosim.conical import conical_nozzle_contour
        from raosim.nozzle_comparison import (
            compare_contours, print_comparison_table, plot_contour_comparison,
        )
        contours = {}
        print("\n  Generating comparison contours...")
        contours['Conical 15°'] = conical_nozzle_contour(Rt, epsilon)
        contours['Bézier bell'] = bell_nozzle_contour(
            Rt, epsilon, length_pct=length_pct)
        try:
            contours['Rao variational'] = bell_nozzle_contour(
                Rt, epsilon, method='rao', length_pct=length_pct)
        except Exception as e:
            print(f"  ⚠ Rao variational failed: {e}")
        results = compare_contours(contours, Pc, Pa, prop.gamma)
        print(print_comparison_table(results))
        if not args.no_plot:
            plot_contour_comparison(contours, results)

    if not args.no_plot:
        plot_nozzle_2d(contour)
        plot_nozzle_3d(contour)

    print("\n  Done.\n")


def run_sweep(args):
    """Parameter sweep mode."""
    _header()

    var, lo, hi, n = args.sweep
    lo, hi, n = float(lo), float(hi), int(n)
    values = np.linspace(lo, hi, n)

    # Propellant
    if args.gamma is not None:
        prop = custom_propellant(args.gamma, args.Mw or 0.022,
                                 args.Tc or 3500, args.eta)
    else:
        prop = get_propellant(args.propellant)

    Pc = args.Pc * 1e5
    Pa = (args.Pa if args.Pa is not None else 101.325) * 1e3
    Rt = args.Rt / 1000.0 if args.Rt else 0.020
    epsilon = args.epsilon or 10.0
    length_pct = args.length_pct

    print(f"  Sweeping '{var}' from {lo} to {hi} ({n} steps)...\n")

    if var == 'epsilon':
        results = sweep_epsilon(values, Pc, Pa, Rt, prop, length_pct)
        x_key = 'epsilon'
    elif var == 'Pc':
        results = sweep_Pc(values, Pa, Rt, epsilon, prop)
        x_key = 'Pc_bar'
    elif var == 'Rt':
        results = sweep_Rt(values, Pc, Pa, epsilon, prop)
        x_key = 'Rt_mm'
    else:
        print(f"  Unknown sweep variable '{var}'. Use: epsilon, Pc, Rt")
        sys.exit(1)


    keys = list(results[0].keys())
    print("  " + "  ".join(f"{k:>10s}" for k in keys))
    for r in results:
        vals = []
        for k in keys:
            v = r[k]
            if v is None:
                vals.append(f"{'N/A':>10s}")
            elif isinstance(v, float):
                vals.append(f"{v:10.4f}")
            else:
                vals.append(f"{v!s:>10s}")
        print("  " + "  ".join(vals))

    if not args.no_plot:
        plot_trade_study(results, x_key,
                         title=f"Trade Study: {var} = [{lo}, {hi}]")


    build_dir, version = create_build_dir()
    output_files: list[str] = []


    import csv as csv_mod
    sweep_csv = build_dir / "sweep_results.csv"
    with open(sweep_csv, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    output_files.append(sweep_csv.name)
    print(f"  → Sweep CSV: {sweep_csv}")

    params = {
        "Propellant": prop.name,
        "Sweep variable": var,
        "Sweep range": f"{lo} → {hi} ({n} steps)",
        "Pc [bar]": f"{Pc / 1e5:.2f}",
        "Pa [kPa]": f"{Pa / 1e3:.3f}",
        "Rt [mm]": f"{Rt * 1000:.2f}",
        "Epsilon (Ae/At)": f"{epsilon:.2f}",
        "Bell length %": f"{length_pct:.1f}",
    }
    meta_path = write_metadata(build_dir, version=version, mode="sweep",
                               params=params, files=output_files)
    output_files.append(meta_path.name)

    print(f"\n  📁 Build v{version:03d}: {build_dir}")
    print("\n  Done.\n")


def run_interactive():
    """Full interactive prompting flow."""
    _header()


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


    Pc_bar = _ask("Chamber pressure Pc [bar]", default=45)
    Pc = Pc_bar * 1e5
    Pa_kPa = _ask("Ambient pressure Pa [kPa] (101.325 = sea level)",
                   default=101.325)
    Pa = Pa_kPa * 1e3
    Rt_mm = _ask("Throat radius Rt [mm]", default=20.0)
    Rt = Rt_mm / 1000.0


    print()
    print("  How to set the expansion ratio?")
    print("    1. Compute from Pc/Pa  (matched expansion)")
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


    print()
    length_pct = _ask("Bell length [% of 15° cone] (60–100)", default=80.0)

    method_choice = _ask_str(
        "Contour method? [bezier / moc]", default="bezier").lower().strip()
    if method_choice not in ('bezier', 'moc'):
        method_choice = 'bezier'

    if method_choice == 'moc':
        print("\n  MOC mode → θ_n will be optimized by the solver")
        contour = bell_nozzle_contour(Rt, epsilon, method='moc',
                                       length_pct=length_pct,
                                       gamma=prop.gamma)
    else:
        tn_default, te_default = lookup_angles(epsilon, length_pct)
        print(f"\n  Rao/NASA table → θ_n = {tn_default:.1f}°, "
              f"θ_e = {te_default:.1f}°")
        override = _ask_str("Use these angles? [Y/n]", default="y").lower()
        if override.startswith("n"):
            theta_n = _ask("θ_n [°]", default=tn_default)
            theta_e = _ask("θ_e [°]", default=te_default)
        else:
            theta_n, theta_e = tn_default, te_default
        contour = bell_nozzle_contour(Rt, epsilon, theta_n, theta_e, length_pct,
                                       gamma=prop.gamma)
    perf = compute_engine_performance(Pc, Pa, Rt, epsilon, prop)
    _print_summary(prop, contour, perf, Pc, Pa, Rt, epsilon)


    print()
    sep = check_separation(contour, Pc, Pa, prop.gamma)
    print(separation_summary(sep))


    do_wp = _ask_str("Compute wall pressure distribution? [Y/n]",
                      default="y").lower()
    if not do_wp.startswith("n"):
        wp = wall_pressure_distribution(contour, Pc, prop.gamma)
        if wp['monotonic']:
            print("  ✓ Wall pressure monotonically decreasing")
        else:
            print(f"  ⚠ Non-monotonic at {len(wp['violation_indices'])} points!")
        plot_wall_pressure(wp)


    do_ch = _ask_str("Generate combustion chamber? [y/N]",
                      default="n").lower()
    if do_ch.startswith("y"):
        L_star = _ask("L* (characteristic length) [m]", default=1.0)
        CR = _ask("Contraction ratio Ac/At", default=2.5)
        ch = chamber_contour(Rt, L_star, CR)
        engine = full_engine_contour(ch, contour)
        print(f"\n  ── Chamber Geometry ────────────────────────────────────")
        print(f"    Chamber radius Rc = {ch['Rc']*1000:.2f} mm")
        print(f"    Cylinder length   = {ch['Lc']*1000:.1f} mm")
        print(f"    Convergent length = {ch['L_conv']*1000:.1f} mm")
        print(f"    Chamber volume    = {ch['V_chamber']*1e6:.2f} cm³")


    do_alt = _ask_str("Show altitude performance map? [y/N]",
                       default="n").lower()
    if do_alt.startswith("y"):
        apm = altitude_performance_map(Pc, Rt, epsilon, prop, contour)
        if apm['h_sep_onset'] is not None:
            print(f"  Separation clears at {apm['h_sep_onset']/1000:.1f} km")
        plot_altitude_performance(apm)


    build_dir, version = create_build_dir()
    output_files: list[str] = []

    print()
    n_csv = int(_ask("Number of CSV points", default=301))
    csv_name = _ask_str("CSV file name", default="rao_nozzle_profile.csv")
    csv_path = export_csv(contour['x'], contour['y'],
                          build_dir / csv_name, n_csv)
    output_files.append(csv_path.name)
    print(f"  → CSV: {csv_path}")

    do_stl = _ask_str("Export STL? [Y/n]", default="y").lower()
    if not do_stl.startswith("n"):
        stl_name = _ask_str("STL file name", default="rao_nozzle.stl")
        n_ang = int(_ask("Angular resolution", default=64))
        stl_path = export_stl(contour['x'], contour['y'],
                              build_dir / stl_name, n_ang)
        output_files.append(stl_path.name)
        print(f"  → STL: {stl_path}")

    # Metadata
    params = {
        "Propellant": prop.name,
        "Pc [bar]": f"{Pc / 1e5:.2f}",
        "Pa [kPa]": f"{Pa / 1e3:.3f}",
        "Rt [mm]": f"{Rt * 1000:.2f}",
        "Epsilon (Ae/At)": f"{epsilon:.2f}",
        "Bell length %": f"{length_pct:.1f}",
        "Theta_n [deg]": f"{contour['theta_n']:.2f}",
        "Theta_e [deg]": f"{contour['theta_e']:.2f}",
        "Gamma": f"{prop.gamma}",
        "Mw [kg/mol]": f"{prop.Mw}",
        "Tc [K]": f"{prop.Tc:.0f}",
        "Eta_Isp": f"{prop.eta_Isp}",
    }
    perf_dict = {
        "Thrust [N]": f"{perf.thrust:.2f}",
        "Thrust [kN]": f"{perf.thrust / 1000:.3f}",
        "Isp [s]": f"{perf.Isp:.1f}",
        "Mass flow [kg/s]": f"{perf.m_dot:.4f}",
        "Ve [m/s]": f"{perf.Ve:.1f}",
        "Exit Mach": f"{perf.Me:.4f}",
        "Exit pressure [Pa]": f"{perf.Pe:.0f}",
        "Cf ideal": f"{perf.Cf_ideal:.4f}",
        "Cf actual": f"{perf.Cf_actual:.4f}",
        "c* [m/s]": f"{perf.c_star:.1f}",
    }
    meta_path = write_metadata(build_dir, version=version, mode="interactive",
                               params=params, performance=perf_dict,
                               files=output_files)
    output_files.append(meta_path.name)

    print(f"\n  📁 Build v{version:03d}: {build_dir}")


    do_plot = _ask_str("Show nozzle plots? [Y/n]", default="y").lower()
    if not do_plot.startswith("n"):
        plot_nozzle_2d(contour, show=True)
        plot_nozzle_3d(contour, show=True)

    print("\n  Done.\n")


def _print_summary(prop, contour, perf, Pc, Pa, Rt, epsilon):
    """Print contour + engine performance summary."""
    print(f"\n  ✓ Propellant: {prop.name}  "
          f"(γ={prop.gamma}, c*={prop.c_star:.0f} m/s)")
    print(f"  ✓ Contour: {len(contour['x'])} pts, "
          f"Ln={contour['Ln']*1000:.1f} mm, "
          f"θ_n={contour['theta_n']:.1f}°, θ_e={contour['theta_e']:.1f}°")
    print(f"    Design status: {contour.get('design_status', 'unknown')}")
    print(f"    Hardware qualified: {contour.get('hardware_qualified', False)}")
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

    if perf.Pe < Pa:
        print(f"    ⚠  Overexpanded (Pe < Pa by {(Pa-perf.Pe)/1000:.2f} kPa)")
    elif perf.Pe > Pa * 1.05:
        print(f"    ⚠  Underexpanded (Pe > Pa by {(perf.Pe-Pa)/1000:.2f} kPa)")
    else:
        print(f"    ✓  Near-matched expansion")


def _print_warnings(warnings: list[str]):
    if not warnings:
        return
    print("\n  ── Design Warnings ─────────────────────────────────────")
    for warning in warnings:
        print(f"    • {warning}")


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.sweep:
        run_sweep(args)
    elif is_batch(args):
        run_batch(args)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
