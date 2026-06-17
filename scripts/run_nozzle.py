"""
run_nozzle.py — the main runner: solve the contour with the
differentiable / MOC backend, then (optionally) add the regen cooling
coils, run the cooling analysis, and export everything.

Pipeline
--------
1. Solve the Rao variational / MOC boundary-value problem
   (``solve_rao_bvp``) on the chosen backend — ``--backend jax`` is the
   differentiable Optimistix-LM path with exact autodiff Jacobians;
   ``--backend numpy`` is the SciPy finite-difference oracle.  The wall
   is built by NASA's BDE-region method-of-characteristics march
   (``wall_method="bde"``), i.e. the real characteristic contour, not
   the chart Bézier.
2. ``--regen`` adds the N regenerative cooling channels (axial, or
   helical "coils" with ``--helix-turns``) wrapping the solved wall.
3. ``--thermal`` runs the coupled Bartz + Sieder-Tate cooling analysis
   and paints the channels with the gas-side wall temperature.

Exports (under ``--out``): ``contour.csv`` (wall polyline),
``profile.png`` (2-D), ``wall.stl``, and with ``--regen`` also
``regen.stl`` + ``regen_3d.png``, plus ``summary.json``.

Run examples
------------
    # INTERACTIVE: run bare (or with -i) and it asks for each dimension
    # (epsilon = expansion ratio, throat radius, length %, ...) showing
    # the [default]; press Enter to accept:
    python scripts/run_nozzle.py
    python scripts/run_nozzle.py -i

    # contour only (differentiable backend), host (full quality):
    PYTHONPATH=. python scripts/run_nozzle.py --backend jax \
        --rt 0.02 --epsilon 10 --length-pct 80 --gamma 1.24 \
        --out builds/run1

    # contour + 80 axial regen channels + thermal colouring:
    PYTHONPATH=. python scripts/run_nozzle.py --backend jax --regen \
        --channels 80 --thermal --pc 7e6 --out builds/run_regen

    # contour + 24 helical coils, 3 turns:
    PYTHONPATH=. python scripts/run_nozzle.py --regen --channels 24 \
        --channel-width 0.0018 --helix-turns 3 --out builds/run_coil

    # AUTO-SIZE the channels from the cooling requirement (coolant flow
    # derived from the cycle); you give the limits, not the geometry:
    PYTHONPATH=. python scripts/run_nozzle.py --regen --auto-size \
        --rt 0.08 --pc 7e6 --mixture-ratio 2.6 --channel-height 0.004 \
        --margin-target 1.2 --dp-budget 300 --out builds/run_sized

    # fast smoke (no LM solve; seed contour only) for a quick check:
    PYTHONPATH=. python scripts/run_nozzle.py --max-nfev 0 --regen \
        --out builds/run_smoke

Note: the full ``--backend jax`` solve (max_nfev ~4000 + weight ladder)
runs for minutes — it is a host job.  ``--max-nfev 0`` evaluates the
seed contour only and is instant (good for wiring/geometry checks).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------- #
# Terminal styling (degrades to plain text when piped or NO_COLOR is set)      #
# --------------------------------------------------------------------------- #
_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(s: str, code: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _COLOR else s


def bold(s): return _c(s, "1")
def dim(s): return _c(s, "2")
def cyan(s): return _c(s, "36")
def orange(s): return _c(s, "38;5;208")
def green(s): return _c(s, "32")
def yellow(s): return _c(s, "33")
def red(s): return _c(s, "31")


_LOGO = r"""
   ██████╗  █████╗  ██████╗      RaoRocketSim
   ██╔══██╗██╔══██╗██╔═══██╗     rocket nozzle design suite
   ██████╔╝███████║██║   ██║     ───────────────────────────
   ██╔══██╗██╔══██║██║   ██║     Rao TOP · MOC · JAX-differentiable
   ██║  ██║██║  ██║╚██████╔╝     regen cooling · 3-D coils
   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝
"""


def print_banner() -> None:
    import re
    for ln in _LOGO.strip("\n").splitlines():
        # art and tagline are separated by a wide (4+ space) gap; the art
        # itself has only 2-space internal gaps, so split on 4+.
        parts = re.split(r" {4,}", ln.rstrip(), maxsplit=1)
        art = parts[0]
        tag = parts[1] if len(parts) > 1 else ""
        print(orange(art) + ("     " + cyan(tag) if tag else ""))
    print()


def print_tags() -> None:
    """Show every run tag (flag), grouped — what the CLI accepts."""
    groups = [
        ("Nozzle", [
            ("--rt", "throat radius [m]"),
            ("--epsilon", "expansion ratio Ae/At"),
            ("--length-pct", "bell length [% of 15° cone]"),
            ("--gamma", "specific-heat ratio"),
            ("--pa-over-p0", "ambient/chamber pressure ratio"),
        ]),
        ("Solver", [
            ("--backend {jax,numpy}", "jax = differentiable LM; numpy = oracle"),
            ("--max-nfev", "LM budget (0 = instant seed contour)"),
            ("--n-control / --n-kernel", "CE / kernel resolution"),
            ("--theta-b-guess", "initial expansion-angle seed [deg]"),
            ("--allow-unconverged", "export even if the 2e-3 gate fails"),
        ]),
        ("Regen coils", [
            ("--regen", "add the cooling channels (the coils)"),
            ("--channels", "channel count"),
            ("--channel-width / --channel-height", "cross-section [m]"),
            ("--wall-thickness", "hot-wall thickness [m]"),
            ("--helix-turns", "0 = axial, >0 = helical coils"),
            ("--coolant / --coolant-mdot", "rp1/methane/lh2/water/ethanol; flow [kg/s]"),
        ]),
        ("Thermal", [
            ("--thermal", "run cooling analysis + colour coils by wall T"),
            ("--pc", "chamber pressure [Pa]"),
            ("--wall-k", "wall conductivity [W/mK]"),
        ]),
        ("Modes / output", [
            ("-i / --interactive", "prompt for everything (default when run bare)"),
            ("--tags", "show this list and exit"),
            ("--no-banner", "suppress the logo"),
            ("--out", "output directory"),
            ("-h / --help", "full argparse help"),
        ]),
    ]
    print(bold("Run tags") + dim("  (pass as flags, or run bare for the interactive interview)"))
    for name, items in groups:
        print("  " + cyan(name))
        for flag, desc in items:
            print("    " + green(f"{flag:<42}") + dim(desc))
    print()

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import raosim.rao_variational as rv
from raosim.rao_variational import RaoSolverConfig
from raosim.regen_geometry import (generate_regen_nozzle, nozzle_wall_surface,
                                   write_stl, _surface_triangles)


def _prompt(label, default, cast=float):
    """Ask for one value, showing the default; Enter accepts it."""
    raw = input(f"  {label} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        return cast(raw)
    except ValueError:
        print(f"    (couldn't parse {raw!r}; keeping {default})")
        return default


def _prompt_bool(label, default):
    raw = input(f"  {label} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    if raw == "":
        return default
    return raw.startswith("y")


def _section(title: str) -> None:
    print("\n" + cyan("▸ " + bold(title)))


def _interactive(args) -> None:
    """Prompt for the design dimensions + options, overriding ``args``."""
    print(dim("Interactive setup — press Enter to accept each [default].  Ctrl-C to abort."))
    _section("Nozzle dimensions")
    args.rt = _prompt("Throat radius Rt [m]", args.rt)
    args.epsilon = _prompt("Expansion ratio  epsilon = Ae/At", args.epsilon)
    args.length_pct = _prompt("Bell length [% of a 15-deg cone]", args.length_pct)
    args.gamma = _prompt("Specific-heat ratio gamma", args.gamma)
    args.pa_over_p0 = _prompt("Ambient/chamber pressure ratio pa/p0", args.pa_over_p0)
    _section("Solver")
    args.backend = _prompt("Backend (jax = differentiable / numpy)", args.backend, str)
    args.max_nfev = _prompt("LM budget max_nfev (0 = instant seed contour)",
                            args.max_nfev, int)
    _section("Regenerative cooling")
    args.regen = _prompt_bool("Add regen cooling channels (the coils)?",
                              bool(args.regen))
    if args.regen:
        args.channel_height = _prompt("Channel height/depth [m]", args.channel_height)
        args.helix_turns = _prompt("Helix turns (0 = axial, >0 = helical coils)",
                                   args.helix_turns)
        args.auto_size = _prompt_bool(
            "Auto-size channel count & width from the cooling requirement?",
            bool(args.auto_size))
        if args.auto_size:
            print(dim("    (you give the requirements; the coolant flow comes "
                      "from the engine cycle, and N & width are solved for)"))
            args.pc = _prompt("Chamber pressure Pc [Pa]", args.pc)
            args.coolant = _prompt("Coolant (rp1/methane/lh2/water/ethanol)",
                                   args.coolant, str)
            args.mixture_ratio = _prompt("Mixture ratio O/F (sets coolant flow)",
                                         args.mixture_ratio)
            args.margin_target = _prompt("Required cooling margin (limit/peak ≥)",
                                         args.margin_target)
            args.wall_temp_limit = _prompt("Max wall temperature [K]", args.wall_temp_limit)
            args.dp_budget = _prompt("Pressure-drop budget [bar]", args.dp_budget)
        else:
            args.channels = _prompt("Channel count", args.channels, int)
            args.channel_width = _prompt("Channel width [m]", args.channel_width)
            args.thermal = _prompt_bool(
                "Run the cooling analysis + colour the coils by wall T?",
                bool(args.thermal))
            if args.thermal:
                args.pc = _prompt("Chamber pressure Pc [Pa]", args.pc)
                args.coolant = _prompt("Coolant (rp1/methane/lh2/water/ethanol)",
                                       args.coolant, str)
                args.coolant_mdot = _prompt("Coolant mass flow [kg/s]", args.coolant_mdot)
    out_raw = input(f"  Output directory [{args.out}]: ").strip()
    if out_raw:
        args.out = Path(out_raw)
    print()


def _solve(args):
    rv.PHYSICS_WEIGHT = 1.0
    cfg = RaoSolverConfig(
        Rt=args.rt, epsilon=args.epsilon, gamma=args.gamma,
        pa_over_p0=args.pa_over_p0, length_pct=args.length_pct,
        n_control=args.n_control, n_kernel=args.n_kernel,
        max_nfev=args.max_nfev,
        solver_backend=args.backend,
        wall_method="bde",
        kernel_d_fraction_max=0.7,
        thetaN_guess_deg=args.theta_b_guess,
    )
    return rv.solve_rao_bvp(cfg)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    # nozzle
    ap.add_argument("--rt", type=float, default=0.020, help="throat radius [m]")
    ap.add_argument("--epsilon", type=float, default=10.0)
    ap.add_argument("--length-pct", type=float, default=80.0)
    ap.add_argument("--gamma", type=float, default=1.24)
    ap.add_argument("--pa-over-p0", type=float, default=0.01)
    # solver
    ap.add_argument("--backend", choices=("jax", "numpy"), default="jax",
                    help="jax = differentiable Optimistix-LM; numpy = SciPy oracle")
    ap.add_argument("--n-control", type=int, default=24)
    ap.add_argument("--n-kernel", type=int, default=24)
    ap.add_argument("--max-nfev", type=int, default=4000,
                    help="LM iteration budget; 0 = seed contour only (instant)")
    ap.add_argument("--theta-b-guess", type=float, default=30.0)
    ap.add_argument("--allow-unconverged", action="store_true")
    # regen (the 'tag' that adds the coils)
    ap.add_argument("--regen", action="store_true",
                    help="add the regenerative cooling channels to the geometry")
    ap.add_argument("--channels", type=int, default=80)
    ap.add_argument("--channel-width", type=float, default=0.0008, help="[m]")
    ap.add_argument("--channel-height", type=float, default=0.0025, help="[m]")
    ap.add_argument("--wall-thickness", type=float, default=0.001, help="[m]")
    ap.add_argument("--helix-turns", type=float, default=0.0,
                    help="0 = axial channels; >0 = helical coils")
    ap.add_argument("--coolant", default="rp1")
    ap.add_argument("--coolant-mdot", type=float, default=10.0, help="[kg/s]")
    # thermal
    ap.add_argument("--thermal", action="store_true",
                    help="run the coupled cooling analysis and colour the coils by wall T")
    ap.add_argument("--pc", type=float, default=7.0e6, help="chamber pressure [Pa]")
    ap.add_argument("--wall-k", type=float, default=350.0, help="wall conductivity [W/mK]")
    # auto-size: SOLVE channel count/width from the cooling requirement
    ap.add_argument("--auto-size", action="store_true",
                    help="size --channels/--channel-width FROM the cooling "
                         "requirement (coolant flow derived from the cycle)")
    ap.add_argument("--margin-target", type=float, default=1.2,
                    help="required cooling margin (wall-temp limit / peak ≥ this)")
    ap.add_argument("--dp-budget", type=float, default=200.0,
                    help="coolant pressure-drop budget [bar]")
    ap.add_argument("--wall-temp-limit", type=float, default=1100.0,
                    help="max allowable gas-side wall temperature [K]")
    ap.add_argument("--mixture-ratio", type=float, default=2.6,
                    help="oxidiser/fuel ratio (sets the cycle coolant flow)")
    ap.add_argument("--cooling-fraction", type=float, default=1.0,
                    help="fraction of the fuel flow routed through the jacket")
    ap.add_argument("--size-objective", choices=("min_dp", "max_margin", "min_channels"),
                    default="min_dp", help="what to optimise among feasible channel designs")
    ap.add_argument("--out", type=Path, default=Path("builds/nozzle_run"))
    ap.add_argument("-i", "--interactive", action="store_true",
                    help="prompt for the dimensions/options instead of using flags")
    ap.add_argument("--tags", action="store_true",
                    help="print every run tag (flag) and exit")
    ap.add_argument("--no-banner", action="store_true", help="suppress the logo")
    args = ap.parse_args()

    bare = len(sys.argv) == 1
    if not args.no_banner:
        print_banner()
    if args.tags:
        print_tags()
        return 0

    # Ask for the inputs when run bare (no flags) or with -i/--interactive.
    if args.interactive or bare:
        print_tags()
        _interactive(args)

    args.out.mkdir(parents=True, exist_ok=True)

    # ---- config summary --------------------------------------------------
    print(cyan("▸ " + bold("Build plan")))
    style = (f"helical {args.helix_turns:g} turns" if args.helix_turns
             else "axial") if args.regen else "—"
    for k, v in [
        ("nozzle", f"Rt={args.rt*1e3:g} mm, eps={args.epsilon:g}, "
                   f"L={args.length_pct:g}%, gamma={args.gamma:g}"),
        ("solver", f"{args.backend}  (max_nfev={args.max_nfev}, "
                   f"n_control={args.n_control}, n_kernel={args.n_kernel})"),
        ("regen", (
            (f"auto-size from requirement (margin≥{args.margin_target:g}, "
             f"Δp≤{args.dp_budget:g} bar, T_wall≤{args.wall_temp_limit:g} K)"
             if args.auto_size else
             f"{args.channels} channels, {style}, "
             f"{args.channel_width*1e3:g}×{args.channel_height*1e3:g} mm")
            if args.regen else dim("off"))),
        ("thermal", (f"on  (Pc={args.pc/1e5:g} bar, {args.coolant}, "
                     + (f"MR={args.mixture_ratio:g} → cycle flow"
                        if args.auto_size else f"{args.coolant_mdot:g} kg/s") + ")"
                     if (args.thermal or args.auto_size) and args.regen
                     else dim("off"))),
        ("output", str(args.out)),
    ]:
        print(f"    {green(k+':'):<14}{v}")

    # ---- 1. solve the contour (differentiable / MOC) -----------------
    print("\n" + cyan("▸ " + bold("Solving contour")) +
          dim("  (Rao variational / MOC BVP)"))
    if args.backend == "jax" and args.max_nfev > 200:
        print(yellow("    note: full JAX LM solve — this runs for minutes."))
    sol = _solve(args)
    r = sol.residuals
    gate = r.max_scaled <= 2e-3
    da = sol.construction_diagnostics.get("design_angles", {})
    badge = green("✓ gate passed") if gate else yellow("● seed / not converged")
    print(f"    max_scaled={r.max_scaled:.3e}   {badge}   "
          f"reliability={sol.reliability.value}")
    print(f"    theta_N={math.degrees(sol.theta_N):.2f}° "
          f"{dim('['+da.get('theta_N_source','?')+']')}   "
          f"theta_E={math.degrees(sol.theta_E):.2f}°   "
          f"Cf={bold('%.4f' % sol.thrust_coefficient)}")
    if not gate and not args.allow_unconverged and args.max_nfev > 0:
        print(yellow("    not converged to the 2e-3 gate; rerun with "
                     "--allow-unconverged or more --max-nfev."), flush=True)

    contour = sol.to_contour_dict(
        Rt=args.rt, epsilon=args.epsilon, length_pct=args.length_pct,
        pa_over_p0=args.pa_over_p0)
    x = np.asarray(contour["x"]); y = np.asarray(contour["y"])

    summary = {
        "backend": args.backend, "Rt": args.rt, "epsilon": args.epsilon,
        "length_pct": args.length_pct, "gamma": args.gamma,
        "max_scaled": float(r.max_scaled), "gate_2e3": bool(gate),
        "reliability": sol.reliability.value,
        "theta_N_deg": math.degrees(sol.theta_N),
        "theta_E_deg": math.degrees(sol.theta_E),
        "Cf": float(sol.thrust_coefficient),
        "exit_radius": float(y[-1]),
    }

    # ---- 2. always export the wall (CSV, 2-D, STL) -------------------
    np.savetxt(args.out / "contour.csv",
               np.column_stack([x, y]), delimiter=",",
               header="x_m,r_m", comments="")
    fig, axp = plt.subplots(figsize=(8, 4))
    axp.plot(x * 1e3, y * 1e3, "b-"); axp.plot(x * 1e3, -y * 1e3, "b-")
    axp.axhline(0, color="k", lw=0.4); axp.set_aspect("equal")
    axp.set_xlabel("x [mm]"); axp.set_ylabel("r [mm]")
    axp.set_title(f"Rao MOC contour  eps={args.epsilon:g} L{args.length_pct:g}%  "
                  f"Cf={sol.thrust_coefficient:.3f}")
    fig.tight_layout(); fig.savefig(args.out / "profile.png", dpi=150); fig.clf()
    wall_verts = nozzle_wall_surface(x, y, n_theta=96)
    write_stl(args.out / "wall.stl", [_surface_triangles(wall_verts)], "rao_wall")
    artifacts = ["contour.csv", "profile.png", "wall.stl"]

    # ---- 2.5 optional channel auto-sizing (solve N/w from requirement) ---
    if args.regen and args.auto_size:
        print("\n" + cyan("▸ " + bold("Sizing channels")) +
              dim("  (from the cooling requirement; coolant flow from the cycle)"))
        from raosim.propellants import custom_propellant
        from raosim.thermal_design import size_cooling_channels
        prop = custom_propellant(gamma=args.gamma, Mw=0.022, Tc=3500.0)
        sized = size_cooling_channels(
            contour, prop, args.pc,
            margin_target=args.margin_target, dp_budget_bar=args.dp_budget,
            wall_temp_limit=args.wall_temp_limit,
            mixture_ratio=args.mixture_ratio,
            cooling_fraction=args.cooling_fraction,
            channel_height=args.channel_height, wall_thickness=args.wall_thickness,
            wall_k=args.wall_k, coolant=args.coolant,
            objective=args.size_objective)
        print(f"    coolant flow (cycle): {sized['mdot_total']:.2f} kg/s total → "
              f"{bold('%.2f kg/s' % sized['mdot_cool'])} coolant "
              f"{dim('(fuel, MR=%g)' % args.mixture_ratio)}")
        if sized["feasible"]:
            print(f"    sized: {bold('%d channels' % sized['channel_count'])} × "
                  f"{bold('%.2f mm' % (sized['channel_width']*1e3))} wide "
                  f"(h={args.channel_height*1e3:g} mm)   "
                  f"{green('margin %.2f ✓' % sized['margin'])}   "
                  f"peak {sized['peak_wall_T']:.0f} K   Δp {sized['pressure_drop_bar']:.0f} bar")
        else:
            print(red(f"    requirement infeasible — best margin {sized['margin']:.2f} "
                      f"at {sized['channel_count']} × {sized['channel_width']*1e3:.2f} mm."))
            print(yellow("    " + sized["diagnosis"]))
            print(dim("    (proceeding with the best-effort design)"))
        # Feed the sized geometry + cycle flow into the rest of the run.
        args.channels = int(sized["channel_count"])
        args.channel_width = float(sized["channel_width"])
        args.coolant_mdot = float(sized["mdot_cool"])
        args.thermal = True            # always colour by the sized cooling state
        summary["channel_sizing"] = {
            "mdot_cool_kg_s": sized["mdot_cool"], "mdot_total_kg_s": sized["mdot_total"],
            "channel_count": sized["channel_count"], "channel_width_m": sized["channel_width"],
            "margin": sized["margin"], "pressure_drop_bar": sized["pressure_drop_bar"],
            "peak_wall_T_K": sized["peak_wall_T"], "feasible": sized["feasible"],
            "diagnosis": sized["diagnosis"],
        }

    # ---- 3. optional cooling analysis --------------------------------
    cooling_result = None
    if args.regen and args.thermal:
        print("\n" + cyan("▸ " + bold("Cooling analysis")) +
              dim("  (Bartz + Sieder-Tate)"))
        from raosim.design import CoolingSpec, MaterialSpec
        from raosim.physics import bartz_heat_flux, regenerative_cooling_analysis
        from raosim.propellants import custom_propellant
        prop = custom_propellant(gamma=args.gamma, Mw=0.022, Tc=3500.0)
        spec = CoolingSpec(method="regenerative", coolant=args.coolant,
                           channel_count=args.channels,
                           channel_width=args.channel_width,
                           channel_height=args.channel_height,
                           coolant_mass_flow=args.coolant_mdot,
                           coolant_inlet_temperature=300.0,
                           max_wall_temperature=args.wall_temp_limit)
        hf = bartz_heat_flux(contour, args.pc, prop, wall_temperature=900.0)
        cooling_result = regenerative_cooling_analysis(
            hf, contour, spec, MaterialSpec(conductivity=args.wall_k),
            args.wall_thickness, prop, args.pc)
        summary["cooling"] = {
            "peak_wall_T_K": cooling_result["peak_gas_side_wall_temperature"],
            "cooling_margin": cooling_result["cooling_margin"],
            "coolant_outlet_T_K": cooling_result["coolant_outlet_temperature"],
            "pressure_drop_bar": cooling_result["coolant_pressure_drop"] / 1e5,
        }
        margin = cooling_result["cooling_margin"]
        mbadge = green(f"margin {margin:.2f} ✓") if margin >= 1.0 else red(f"margin {margin:.2f} ✗")
        print(f"    peak wall {bold('%.0f K' % cooling_result['peak_gas_side_wall_temperature'])}   "
              f"{mbadge}   Δp {cooling_result['coolant_pressure_drop']/1e5:.1f} bar   "
              f"coolant out {cooling_result['coolant_outlet_temperature']:.0f} K")

    # ---- 4. optional regen geometry (the coils) ----------------------
    if args.regen:
        print("\n" + cyan("▸ " + bold("Regen geometry")) + dim("  (the coils)"))
        from raosim.design import CoolingSpec
        spec = CoolingSpec(method="regenerative", coolant=args.coolant,
                           channel_count=args.channels,
                           channel_width=args.channel_width,
                           channel_height=args.channel_height)
        reg = generate_regen_nozzle(
            contour, spec, args.wall_thickness,
            helix_turns=args.helix_turns,
            stl_path=args.out / "regen.stl",
            png_path=args.out / "regen_3d.png",
            cooling_result=cooling_result)
        s = reg["summary"]
        style = f"helix {s['helix_turns']:.1f} turns" if s["helix_turns"] else "axial"
        summary["regen"] = {**s, "style": style, "n_triangles": reg["n_triangles"]}
        fit = green("fit ✓") if s["channels_fit"] else red("OVERFLOWS ✗")
        print(f"    {s['n_channels']} channels ({style})   {fit}   "
              f"{reg['n_triangles']:,} triangles")
        artifacts += ["regen.stl", "regen_3d.png"]
        if not s["channels_fit"]:
            print(red("    channels exceed the throat circumference; "
                      "reduce --channels or --channel-width."))

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    artifacts.append("summary.json")

    # ---- results panel ----------------------------------------------------
    print("\n" + green("▸ " + bold("Done")) + f"  →  {bold(str(args.out))}/")
    for a in artifacts:
        print(f"    {dim('•')} {a}")
    print(dim("\n  Preliminary design geometry — not hardware-qualified "
              "(needs CFD, thermal/structural review, hot-fire)."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
