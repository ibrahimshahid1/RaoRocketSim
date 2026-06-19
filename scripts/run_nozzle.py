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

Exports (under ``--out``): ``contour.csv`` (hot-gas wall polyline),
``profile.png`` (2-D), a closed ``wall.stl``, and optionally ``wall.step``
(``--cad step``).  With ``--regen`` it also writes the channel-visualization
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
            ("--coolant-outlet-pressure / --injector-pressure-drop",
             "absolute jacket outlet pressure / injector loss [Pa]"),
        ]),
        ("Thermal", [
            ("--thermal", "run cooling analysis + colour coils by wall T"),
            ("--pc", "chamber pressure [Pa]"),
            ("--material / --list-materials", "wall metal from the catalog "
             "(k, temp limit, structural E/a/v)"),
            ("--jacket-material", "optional stronger outer closeout alloy"),
            ("--wall-k / --wall-temp-limit", "override the material's k / T limit"),
            ("--auto-size / --size-wall", "size channels (cooling) / wall+channels "
             "(thermal + SP-125 structural)"),
            ("--margin-target / --structural-fos / --required-cycles / --dp-budget",
             "sizing requirements (thermal / stress / fatigue N_f / Δp)"),
        ]),
        ("Visualisation", [
            ("--flowfield", "render the steady MOC Mach/temperature field"),
            ("--animate {march,particles,both}", "save a flow animation GIF"),
            ("--chamber-temp", "chamber T [K] for the temperature panel"),
        ]),
        ("Modes / output", [
            ("-i / --interactive", "prompt for everything (default when run bare)"),
            ("--show", "pop up the plots in a live window (auto for interactive)"),
            ("--tags", "show this list and exit"),
            ("--no-banner", "suppress the logo"),
            ("--out", "output directory"),
            ("--cad {none,step,ipt,both}", "export a STEP solid / IPT conversion manifest"),
            ("--require-brep", "fail instead of using a triangle-faceted STEP fallback"),
            ("--regen-brep", "export one full-N liner+ribs+jacket STEP solid"),
            ("--regen-manifolds", "also cut plenums and area-sized radial ports"),
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
# Pop up live windows for interactive / --show runs, but ONLY when
# attached to a real terminal — otherwise (piped, headless, CI) render
# to files (Agg) so plt.show() never hangs waiting for a display.
# Decided before pyplot is imported.
_INTERACTIVE_RUN = (
    len(sys.argv) == 1 or "-i" in sys.argv or "--interactive" in sys.argv
)
_WANT_WINDOWS = (("--show" in sys.argv or _INTERACTIVE_RUN)
                 and sys.stdout.isatty())
if not _WANT_WINDOWS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import raosim.rao_variational as rv
from raosim.export import (
    export_step,
    export_stl,
    package_ipt_request,
    step_representation,
)
from raosim.materials import get_material, material_names, material_table
from raosim.rao_variational import RaoSolverConfig
from raosim.regen_geometry import generate_regen_nozzle


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
    args.flowfield = _prompt_bool(
        "Render the steady flow field (Mach + temperature)?",
        bool(args.flowfield))
    if _prompt_bool("Animate the flow (MOC march + particle advection)?",
                    args.animate is not None):
        args.animate = _prompt(
            "  which? (march / particles / both)", args.animate or "both", str)
    _section("Regenerative cooling")
    args.regen = _prompt_bool("Add regen cooling channels (the coils)?",
                              bool(args.regen))
    if args.regen:
        print(dim("    Wall materials: " + ", ".join(material_names())))
        args.material = _prompt(
            "Wall material (sets k + temp limit + structural; Enter for none)",
            args.material or "", str) or None
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
    ap.add_argument("--channel-height-min", type=float, default=None,
                    help="minimum channel depth searched by --size-wall [m]")
    ap.add_argument("--channel-height-max", type=float, default=None,
                    help="maximum channel depth searched by --size-wall [m]")
    ap.add_argument("--channel-height-steps", type=int, default=3,
                    help="number of channel-depth candidates for --size-wall")
    ap.add_argument("--wall-thickness", type=float, default=0.001, help="[m]")
    ap.add_argument("--helix-turns", type=float, default=0.0,
                    help="0 = axial channels; >0 = helical coils")
    ap.add_argument("--coolant", default="rp1")
    ap.add_argument("--coolant-mdot", type=float, default=10.0, help="[kg/s]")
    ap.add_argument("--coolant-outlet-pressure", type=float, default=None,
                    help="absolute coolant pressure at jacket outlet [Pa]; "
                         "default Pc + injector pressure drop")
    ap.add_argument("--injector-pressure-drop", type=float, default=0.0,
                    help="injector pressure drop after the jacket outlet [Pa]")
    # thermal
    ap.add_argument("--thermal", action="store_true",
                    help="run the coupled cooling analysis and colour the coils by wall T")
    ap.add_argument("--pc", type=float, default=7.0e6, help="chamber pressure [Pa]")
    ap.add_argument("--material", default=None,
                    help="wall material from the catalog (e.g. grcop-84, "
                         "narloy-z, inconel718, 316l); sets k + temp limit + "
                         "structural props.  --list-materials to see all.")
    ap.add_argument("--list-materials", action="store_true",
                    help="print the materials catalog and exit")
    # --wall-k / --wall-temp-limit default to None: when a --material is
    # given they come from the catalog; either flag still overrides it.
    ap.add_argument("--wall-k", type=float, default=None,
                    help="wall conductivity [W/mK] (overrides the material; "
                         "default 350 when no --material)")
    # auto-size: SOLVE channel count/width from the cooling requirement
    ap.add_argument("--auto-size", action="store_true",
                    help="size --channels/--channel-width FROM the cooling "
                         "requirement (coolant flow derived from the cycle)")
    ap.add_argument("--margin-target", type=float, default=1.2,
                    help="required cooling margin (wall-temp limit / peak ≥ this)")
    ap.add_argument("--dp-budget", type=float, default=200.0,
                    help="coolant pressure-drop budget [bar]")
    ap.add_argument("--wall-temp-limit", type=float, default=None,
                    help="max allowable gas-side wall temperature [K] "
                         "(overrides the material; default 1100 when no "
                         "--material)")
    ap.add_argument("--mixture-ratio", type=float, default=2.6,
                    help="oxidiser/fuel ratio (sets the cycle coolant flow)")
    ap.add_argument("--cooling-fraction", type=float, default=1.0,
                    help="fraction of the fuel flow routed through the jacket")
    ap.add_argument("--size-objective", choices=("min_dp", "max_margin", "min_channels"),
                    default="min_dp", help="what to optimise among feasible channel designs")
    # joint wall+channel sizing: co-size t_hot AND the channels against the
    # thermal AND SP-125 eq.4-31 structural limits (needs --material).
    ap.add_argument("--size-wall", action="store_true",
                    help="co-size hot-wall thickness + channels vs thermal AND "
                         "structural limits (needs --material)")
    ap.add_argument("--structural-fos", type=float, default=1.0,
                    help="structural factor of safety yield/combined-stress "
                         "(copper liners run near yield / LCF; default 1.0)")
    ap.add_argument("--required-cycles", type=float, default=100.0,
                    help="required thermal-cycle life (Coffin-Manson N_f screen)")
    ap.add_argument("--life-fos", type=float, default=4.0,
                    help="factor of safety on cyclic life (N_f ≥ required × this)")
    ap.add_argument("--t-hot-max", type=float, default=0.003,
                    help="max hot-wall thickness to search [m]")
    ap.add_argument("--buckling-fos", type=float, default=1.0,
                    help="factor of safety for SP-125 4-29 and external-pressure buckling")
    ap.add_argument("--buckling-tangent-modulus-fraction", type=float, default=0.10,
                    help="screening Et/E and Ec/E used by SP-125 eq.4-29 "
                         "until sourced hot-wall tangent-modulus data are supplied")
    ap.add_argument("--gate-sp125-tube-buckling", action="store_true",
                    help="let the equivalent-tube SP-125 eq.4-29 screen gate "
                         "a milled-channel wall (off by default)")
    ap.add_argument("--flowfield", action="store_true",
                    help="render the steady MOC Mach/temperature flow field")
    ap.add_argument("--animate", choices=("march", "particles", "both"),
                    default=None,
                    help="save an animation GIF: 'march' (MOC net building), "
                         "'particles' (tracer advection), or 'both'")
    ap.add_argument("--chamber-temp", type=float, default=3500.0,
                    help="chamber temperature [K] for the flow-field temperature panel")
    ap.add_argument("--out", type=Path, default=Path("builds/nozzle_run"))
    ap.add_argument("--cad", choices=("none", "step", "ipt", "both"), default="none",
                    help="optional solid CAD export; IPT writes an Inventor "
                         "conversion manifest around the authoritative STEP")
    ap.add_argument("--require-brep", action="store_true",
                    help="require CadQuery/OpenCascade true B-rep STEP output; "
                         "do not accept the faceted AP214 fallback")
    ap.add_argument("--regen-brep", action="store_true",
                    help="export a full-N one-solid regenerative wall STEP "
                         "(patterned positive ribs; requires CadQuery)")
    ap.add_argument("--regen-manifolds", action="store_true",
                    help="with --regen-brep, cut annular plenums and radial "
                         "ports into the one-solid wall")
    ap.add_argument("--regen-cad-sections", type=int, default=24,
                    help="axial loft sections used by --regen-brep")
    ap.add_argument("--regen-ports-per-manifold", type=int, default=4,
                    help="radial ports on each plenum for --regen-manifolds")
    ap.add_argument("--regen-port-area-ratio", type=float, default=1.0,
                    help="total port area / total channel area")
    ap.add_argument("--regen-port-diameter", type=float, default=None,
                    help="override each manifold port diameter [m]")
    ap.add_argument("--jacket-material", default=None,
                    help="outer-jacket material from the catalog for --size-wall "
                         "(defaults to the liner); e.g. inconel718 over a copper liner")
    ap.add_argument("-i", "--interactive", action="store_true",
                    help="prompt for the dimensions/options instead of using flags")
    ap.add_argument("--tags", action="store_true",
                    help="print every run tag (flag) and exit")
    ap.add_argument("--no-banner", action="store_true", help="suppress the logo")
    ap.add_argument("--show", action="store_true",
                    help="pop up the flow-field / animation in a live window "
                         "(on by default for interactive runs)")
    args = ap.parse_args()
    if args.regen_manifolds:
        args.regen_brep = True
    if args.regen_brep and not args.regen:
        ap.error("--regen-brep requires --regen")
    if args.regen_brep and args.cad == "none":
        ap.error("--regen-brep requires --cad step, ipt, or both")
    # Live windows for interactive/--show; saved files otherwise.
    show = bool(_WANT_WINDOWS)

    bare = len(sys.argv) == 1
    if not args.no_banner:
        print_banner()
    if args.tags:
        print_tags()
        return 0
    if args.list_materials:
        print(cyan("▸ " + bold("Materials catalog")) +
              dim("  (select with --material <name>)"))
        print(material_table())
        return 0

    # Ask for the inputs when run bare (no flags) or with -i/--interactive.
    if args.interactive or bare:
        print_tags()
        _interactive(args)

    # ---- resolve the wall material -------------------------------------
    # A --material populates conductivity + the gas-side temperature limit
    # (and the structural E/a/v); --wall-k / --wall-temp-limit still win
    # when given explicitly.  Without either, keep the prior defaults.
    mat = None
    if args.material:
        try:
            mat = get_material(args.material)
        except KeyError as exc:
            print(red(f"    {exc}"))
            return 2
    if args.wall_k is None:
        args.wall_k = mat.conductivity if mat else 350.0
    if args.wall_temp_limit is None:
        args.wall_temp_limit = mat.max_temperature if mat else 1100.0
    args._material = mat

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
        ("material", (f"{mat.name}  (k={args.wall_k:g} W/mK, "
                      f"T≤{args.wall_temp_limit:g} K, "
                      f"{mat.category.replace('_', ' ')})" if mat else
                      dim(f"unspecified  (k={args.wall_k:g} W/mK, "
                          f"T≤{args.wall_temp_limit:g} K)"))),
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
    summary["material"] = {
        "name": mat.name if mat else None,
        "category": mat.category if mat else None,
        "conductivity_W_mK": args.wall_k,
        "max_wall_temperature_K": args.wall_temp_limit,
        "source": mat.source if mat else None,
    }

    # ---- 2. export the contour reference (solid wall follows sizing) --
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
    artifacts = ["contour.csv", "profile.png"]

    # ---- optional steady flow-field render ---------------------------
    if args.flowfield:
        print("\n" + cyan("▸ " + bold("Flow field")) +
              dim("  (MOC Mach + temperature, characteristics, streamlines)"))
        try:
            from raosim.flow_viz import plot_flowfield
            fig = plot_flowfield(sol, gamma=args.gamma, Tc=args.chamber_temp,
                                 save_path=args.out / "flowfield.png", show=show)
            if not show:
                fig.clf()
            artifacts.append("flowfield.png")
            print(green("    wrote flowfield.png")
                  + (dim("  (window)") if show else ""))
        except Exception as exc:
            print(yellow(f"    flow field skipped: {exc}"))

    # ---- optional animations (saved as GIF; show=True pops a window) --
    if args.animate:
        print("\n" + cyan("▸ " + bold("Animation")) +
              dim("  (GIF; use show=True in Python for a live window)"))
        if args.animate in ("march", "both"):
            try:
                from raosim.flow_viz import animate_moc_march
                animate_moc_march(sol, gamma=args.gamma,
                                  save_path=args.out / "anim_moc_march.gif",
                                  fps=8, show=show)
                artifacts.append("anim_moc_march.gif")
                print(green("    wrote anim_moc_march.gif")
                      + (dim("  (window)") if show else ""))
            except Exception as exc:
                print(yellow(f"    MOC march skipped: {exc}"))
        if args.animate in ("particles", "both"):
            try:
                from raosim.flow_viz import animate_particles
                animate_particles(sol, gamma=args.gamma, Tc=args.chamber_temp,
                                  save_path=args.out / "anim_particles.gif",
                                  fps=25, show=show)
                artifacts.append("anim_particles.gif")
                print(green("    wrote anim_particles.gif")
                      + (dim("  (window)") if show else ""))
            except Exception as exc:
                print(yellow(f"    particles skipped: {exc}"))

    # ---- 2.45 optional JOINT wall+channel sizing (t_hot + N + w) ---------
    if args.regen and args.size_wall:
        if args._material is None:
            print(red("\n    --size-wall needs --material for the structural "
                      "properties (E/α/ν/yield); see --list-materials."))
            return 2
        print("\n" + cyan("▸ " + bold("Sizing wall + channels")) +
              dim("  (thermal + SP-125 eq.4-31 structural; coolant from the cycle)"))
        from raosim.design import MaterialSpec
        from raosim.propellants import custom_propellant
        from raosim.thermal_design import joint_wall_channel_design
        prop = custom_propellant(gamma=args.gamma, Mw=0.022, Tc=3500.0)
        obj = args.size_objective if args.size_objective == "min_dp" else "min_mass"
        sizing_material = MaterialSpec.from_catalog(args._material.name)
        sizing_material.conductivity = float(args.wall_k)
        sizing_material.max_temperature = float(args.wall_temp_limit)
        jd = joint_wall_channel_design(
            contour, prop, args.pc, material=sizing_material,
            mixture_ratio=args.mixture_ratio, cooling_fraction=args.cooling_fraction,
            coolant=args.coolant, thermal_margin=args.margin_target,
            structural_fos=args.structural_fos, required_cycles=args.required_cycles,
            life_fos=args.life_fos, dp_budget_bar=args.dp_budget,
            helix_turns=args.helix_turns, channel_height=args.channel_height,
            channel_height_min=(
                args.channel_height_min
                if args.channel_height_min is not None
                else 0.60 * args.channel_height
            ),
            channel_height_max=(
                args.channel_height_max
                if args.channel_height_max is not None
                else 1.80 * args.channel_height
            ),
            n_height=args.channel_height_steps,
            t_hot_max=args.t_hot_max, objective=obj,
            coolant_outlet_pressure=args.coolant_outlet_pressure,
            injector_pressure_drop=args.injector_pressure_drop)
        print(f"    coolant flow (cycle): {jd['mdot_total']:.2f} kg/s total → "
              f"{bold('%.2f kg/s' % jd['mdot_cool'])} coolant "
              f"{dim('(fuel, MR=%g)' % args.mixture_ratio)}")
        if jd["channel_count"] is None:
            print(red("    no channel geometry fits the throat; reduce the count."))
            return 2
        b = jd["band"]
        if jd["feasible"]:
            if jd["fatigue_status"] == "design_qualified_gate":
                fatigue_text = green("N_f %.0f ✓" % jd["fatigue_cycles"])
            elif jd["fatigue_status"] == "screening_only_not_gating":
                fatigue_text = yellow("N_f %.0f (screen only)" % jd["fatigue_cycles"])
            else:
                fatigue_text = dim("N_f not evaluated")
            print(f"    sized: {bold('t_hot %.2f mm' % (jd['t_hot']*1e3))}, "
                  f"{bold('%d channels' % jd['channel_count'])} × "
                  f"{jd['channel_width']*1e3:.2f} mm   "
                  f"{green('thermal %.2f ✓' % jd['thermal_margin'])} "
                  f"{green('σ %.2f ✓' % jd['structural_margin'])} "
                  f"{fatigue_text}   "
                  f"peak {jd['peak_wall_T']:.0f} K   Δp {jd['pressure_drop_bar']:.0f} bar   "
                  f"mass {jd['mass_kg']:.1f} kg")
            if b:
                fat = (f", fatigue ∈ [{b.get('t_fatigue_lo', float('nan'))*1e3:.2f}, "
                       f"{b.get('t_fatigue_hi', float('nan'))*1e3:.2f}]"
                       if "t_fatigue_lo" in b else "")
                print(dim(f"    feasible hot-wall band: t_hot ∈ "
                          f"[{b['feasible_lo']*1e3:.2f}, {b['feasible_hi']*1e3:.2f}] mm "
                          f"(mfg ≥ {b['t_manufacturing']*1e3:.2f}, structural ≥ "
                          f"{b['t_structural_lo']*1e3:.2f}{fat}, thermal ≤ "
                          f"{b['t_thermal_max']*1e3:.2f})"))
            if jd["fatigue_cycles"] is not None:
                print(dim(
                    f"    fatigue: N_f ≈ {jd['fatigue_cycles']:.0f} cycles at "
                    f"Δε {jd['strain_range']*100:.2f}% "
                    f"[{jd['fatigue_status']}; source: {jd['fatigue_source']}]"
                ))
            else:
                print(dim(
                    "    fatigue: not evaluated; SP-125 identifies thermal LCF "
                    "as governing, but this material has no sourced strain-life coefficients"
                ))
        else:
            fatigue_report = (
                f", N_f {jd['fatigue_cycles']:.0f}"
                if jd["fatigue_cycles"] is not None else ", N_f not evaluated"
            )
            print(red(f"    requirement infeasible — best thermal {jd['thermal_margin']:.2f}, "
                      f"σ {jd['structural_margin']:.2f}{fatigue_report} "
                      f"at t_hot {jd['t_hot']*1e3:.2f} mm."))
            print(yellow("    " + jd["diagnosis"]))
            print(dim("    (proceeding with the best-effort design)"))
        args.channels = int(jd["channel_count"])
        args.channel_width = float(jd["channel_width"])
        args.channel_height = float(jd["channel_height"])
        args.wall_thickness = float(jd["t_hot"])
        args.coolant_mdot = float(jd["mdot_cool"])
        args.thermal = True
        summary["wall_sizing"] = {
            "material": jd["material"], "t_hot_m": jd["t_hot"],
            "channel_count": jd["channel_count"], "channel_width_m": jd["channel_width"],
            "channel_height_m": jd["channel_height"],
            "thermal_margin": jd["thermal_margin"], "structural_margin": jd["structural_margin"],
            "combined_stress_MPa": jd["combined_stress_MPa"],
            "fatigue_cycles": jd["fatigue_cycles"], "strain_range": jd["strain_range"],
            "fatigue_status": jd["fatigue_status"],
            "fatigue_source": jd["fatigue_source"],
            "fatigue_gates_feasibility": jd["fatigue_gates_feasibility"],
            "required_cycles": args.required_cycles, "life_fos": args.life_fos,
            "pressure_drop_bar": jd["pressure_drop_bar"], "peak_wall_T_K": jd["peak_wall_T"],
            "mass_kg": jd["mass_kg"], "feasible": jd["feasible"],
            "max_liner_pressure_differential_bar":
                jd["max_liner_pressure_differential_bar"],
            "coolant_pressure_boundary_source":
                jd["coolant_pressure_boundary_source"],
            "band": jd["band"], "diagnosis": jd["diagnosis"],
        }

        # Refine the chosen channels into a VARIABLE throat→exit wall: size
        # t_hot(x) (thinnest that holds the SP-125 eq.4-31 stress) + the
        # jacket t_jacket(x) (outer-shell coolant hoop) station by station,
        # so the exported geometry is the manufacturable wall, not one
        # uniform thickness.
        from raosim.thermal_design import size_wall_profile
        prof = size_wall_profile(
            contour, prop, args.pc, material=sizing_material,
            jacket_material=args.jacket_material,
            channel_count=args.channels, channel_width=args.channel_width,
            channel_height=args.channel_height, mixture_ratio=args.mixture_ratio,
            cooling_fraction=args.cooling_fraction, coolant=args.coolant,
            thermal_margin=args.margin_target, structural_fos=args.structural_fos,
            helix_turns=args.helix_turns, t_hot_max=args.t_hot_max,
            channel_height_min=args.channel_height_min,
            channel_height_max=args.channel_height_max,
            n_channel_height=args.channel_height_steps,
            dp_budget_bar=args.dp_budget,
            buckling_fos=args.buckling_fos,
            buckling_tangent_modulus_fraction=
                args.buckling_tangent_modulus_fraction,
            gate_sp125_429=args.gate_sp125_tube_buckling,
            coolant_outlet_pressure=args.coolant_outlet_pressure,
            injector_pressure_drop=args.injector_pressure_drop)
        args._wall_profile = prof["profile"]
        jck = (f" + {prof['jacket_material']} jacket" if args.jacket_material
               else " + jacket")
        vbadge = green("✓") if prof["feasible"] else yellow("partial (see per-station)")
        print(f"    variable wall: t_hot {bold('%.2f–%.2f mm' % (prof['t_hot_min_mm'], prof['t_hot_max_mm']))} "
              f"(throat {prof['t_hot_throat_mm']:.2f}){jck} "
              f"h {prof['channel_height_min_mm']:.2f}–{prof['channel_height_max_mm']:.2f} mm "
              f"{prof['t_jacket_min_mm']:.2f}–{prof['t_jacket_max_mm']:.2f} mm   "
              f"{vbadge}   {dim('min σ %.2f · mass %.1f kg' % (prof['min_structural_margin'], prof['mass_kg']))}")
        summary["wall_profile"] = {
            "t_hot_throat_mm": prof["t_hot_throat_mm"],
            "t_hot_range_mm": [prof["t_hot_min_mm"], prof["t_hot_max_mm"]],
            "t_jacket_range_mm": [prof["t_jacket_min_mm"], prof["t_jacket_max_mm"]],
            "channel_height_range_mm": [
                prof["channel_height_min_mm"],
                prof["channel_height_max_mm"],
            ],
            "jacket_material": prof["jacket_material"],
            "min_structural_margin": prof["min_structural_margin"],
            "min_jacket_margin": prof["min_jacket_margin"],
            "min_sp125_429_margin": prof["min_sp125_429_margin"],
            "min_external_buckling_margin":
                prof["min_external_buckling_margin"],
            "buckling_data_status": prof["buckling_data_status"],
            "sp125_429_geometry_status": prof["sp125_429_geometry_status"],
            "sp125_429_gates_feasibility":
                prof["sp125_429_gates_feasibility"],
            "thermal_feasible": prof["thermal_feasible"],
            "liner_mass_kg": prof["liner_mass_kg"],
            "jacket_mass_kg": prof["jacket_mass_kg"],
            "feasible": prof["feasible"],
        }

    # ---- 2.5 optional channel auto-sizing (solve N/w from requirement) ---
    if args.regen and args.auto_size and not args.size_wall:
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
            helix_turns=args.helix_turns, objective=args.size_objective)
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
        from raosim.physics import (
            bartz_heat_flux,
            coaxial_shell_wall_stress_profile,
            regenerative_cooling_analysis,
        )
        from raosim.propellants import custom_propellant
        prop = custom_propellant(gamma=args.gamma, Mw=0.022, Tc=3500.0)
        spec = CoolingSpec(method="regenerative", coolant=args.coolant,
                           channel_count=args.channels,
                           channel_width=args.channel_width,
                           channel_height=args.channel_height,
                           coolant_mass_flow=args.coolant_mdot,
                           coolant_inlet_temperature=300.0,
                           coolant_outlet_pressure=args.coolant_outlet_pressure,
                           injector_pressure_drop=args.injector_pressure_drop,
                           max_wall_temperature=args.wall_temp_limit)
        # Material from the catalog (k, S_y, T_max, E, a, v) so the cooling
        # and structural screens see one coherent metal; --wall-k still wins.
        material = (MaterialSpec.from_catalog(args._material.name)
                    if args._material else MaterialSpec(conductivity=args.wall_k))
        material.conductivity = args.wall_k
        hf = bartz_heat_flux(contour, args.pc, prop, wall_temperature=900.0)
        analysis_wall_profile = getattr(args, "_wall_profile", None)
        cooling_result = regenerative_cooling_analysis(
            hf, contour, spec, material, args.wall_thickness, prop, args.pc,
            helix_turns=args.helix_turns,
            wall_profile=analysis_wall_profile,
            coolant_outlet_pressure=args.coolant_outlet_pressure,
            injector_pressure_drop=args.injector_pressure_drop)
        pf = cooling_result.get("passage_length_factor", 1.0)
        summary["cooling"] = {
            "peak_wall_T_K": cooling_result["peak_gas_side_wall_temperature"],
            "cooling_margin": cooling_result["cooling_margin"],
            "coolant_outlet_T_K": cooling_result["coolant_outlet_temperature"],
            "pressure_drop_bar": cooling_result["coolant_pressure_drop"] / 1e5,
            "helix_turns": cooling_result.get("helix_turns", 0.0),
            "passage_length_factor": pf,
            "coolant_inlet_pressure_Pa": cooling_result["coolant_inlet_pressure"],
            "coolant_outlet_pressure_Pa": cooling_result["coolant_outlet_pressure"],
            "coolant_pressure_boundary_source":
                cooling_result["coolant_pressure_boundary_source"],
        }
        margin = cooling_result["cooling_margin"]
        mbadge = green(f"margin {margin:.2f} ✓") if margin >= 1.0 else red(f"margin {margin:.2f} ✗")
        dp_note = dim(f"  (helix ×{pf:.2f} path)") if args.helix_turns else ""
        print(f"    peak wall {bold('%.0f K' % cooling_result['peak_gas_side_wall_temperature'])}   "
              f"{mbadge}   Δp {cooling_result['coolant_pressure_drop']/1e5:.1f} bar{dp_note}   "
              f"coolant out {cooling_result['coolant_outlet_temperature']:.0f} K")

        # ---- structural screen: SP-125 eq. 4-31 combined wall stress -----
        # Needs the elastic/thermal properties, which only a catalog
        # --material supplies.  The thin liner carries the station-wise
        # COOLANT-GAS differential from the hydraulic and gas-pressure
        # marches, not a fixed fraction of chamber pressure.
        if material.elastic_modulus:
            stress = coaxial_shell_wall_stress_profile(
                pressure_differential=cooling_result["liner_pressure_differential"],
                inner_radius=contour["y"],
                wall_thickness=(
                    analysis_wall_profile.t_hot
                    if analysis_wall_profile is not None
                    else args.wall_thickness
                ),
                heat_flux=cooling_result["q"],
                elastic_modulus=material.elastic_modulus,
                thermal_expansion=material.thermal_expansion,
                poisson_ratio=material.poisson_ratio, conductivity=args.wall_k,
                yield_strength=material.yield_strength)
            sm = stress["stress_margin"]
            txt = f"σ-margin {sm:.2f}"
            # Copper liners run near yield (LCF), so ≥1.0 is acceptable.
            sbadge = green(txt + " ✓") if sm >= 1.0 else (
                yellow(txt + " (near yield/LCF)") if sm >= 0.8 else red(txt + " ✗"))
            print(f"    wall stress {bold('%.0f MPa' % (stress['combined_stress']/1e6))} "
                  f"{dim('(thermal %.0f + pressure %.0f)' % (stress['thermal_stress']/1e6, stress['pressure_stress']/1e6))}   "
                  f"{sbadge}   {dim('[SP-125 eq.4-31 station-wise, S_y=%.0f MPa]' % (material.yield_strength/1e6))}")
            summary["structural"] = {
                "combined_stress_MPa": stress["combined_stress"] / 1e6,
                "thermal_stress_MPa": stress["thermal_stress"] / 1e6,
                "pressure_stress_MPa": stress["pressure_stress"] / 1e6,
                "yield_strength_MPa": material.yield_strength / 1e6,
                "stress_margin": sm, "model": stress["model"],
                "governing_index": stress["governing_index"],
                "max_liner_pressure_differential_bar":
                    float(max(cooling_result["liner_pressure_differential"]) / 1e5),
            }

    # ---- 3.8 solid wall CAD, using the final sized thickness ----------
    # Unlike the regen visualization below, this is a closed material body:
    # inner hot-gas surface + normal-offset outer surface + annular end caps.
    wall_profile = None
    wall_thickness_geometry = args.wall_thickness
    sized_wall_profile = getattr(args, "_wall_profile", None)
    if args.regen:
        from raosim.regen_profile import RegenWallProfile
        # Prefer the VARIABLE profile from --size-wall (t_hot(x) + jacket);
        # otherwise build a uniform one from the scalar thickness.
        wall_profile = sized_wall_profile
        if wall_profile is None:
            wall_profile = RegenWallProfile.uniform(
                contour,
                channel_count=args.channels,
                channel_width=args.channel_width,
                channel_height=args.channel_height,
                t_hot=args.wall_thickness,
                helix_turns=args.helix_turns,
            )
        wall_thickness_geometry = wall_profile.t_hot
    wall_path = export_stl(
        x, y, args.out / "wall.stl", n_angular=96,
        wall_thickness=wall_thickness_geometry,
    )
    artifacts.append(wall_path.name)
    summary["wall_geometry"] = {
        "uniform_seed_thickness_m": float(args.wall_thickness),
        "t_hot_range_m": [
            float(np.min(np.asarray(wall_thickness_geometry))),
            float(np.max(np.asarray(wall_thickness_geometry))),
        ],
        "offset": "surface_normal",
        "stl": "closed_solid_triangle_mesh",
        "wall_scope": "liner_base_only_no_channel_ribs_or_jacket",
    }

    # A station-wise --size-wall result also defines a separately sized
    # closeout jacket.  Export it as its own closed shell; the ribs/channel
    # Boolean geometry between liner and jacket is still visualization-only.
    jacket_inner_x = jacket_inner_r = None
    if sized_wall_profile is not None:
        from raosim.regen_profile import normal_offset_contour
        jacket_inner_x, jacket_inner_r = normal_offset_contour(
            x, y,
            sized_wall_profile.t_hot + sized_wall_profile.channel_height,
        )
        jacket_stl = export_stl(
            jacket_inner_x, jacket_inner_r, args.out / "jacket.stl",
            n_angular=96, wall_thickness=sized_wall_profile.t_jacket,
        )
        artifacts.append(jacket_stl.name)
        summary["wall_geometry"]["jacket_stl"] = "closed_solid_triangle_mesh"
        summary["wall_geometry"]["t_jacket_range_m"] = [
            float(np.min(sized_wall_profile.t_jacket)),
            float(np.max(sized_wall_profile.t_jacket)),
        ]

    step_path = None
    regen_step_path = None
    if args.cad in ("step", "ipt", "both"):
        try:
            step_path = export_step(
                x, y, args.out / "wall.step", n_angular=96,
                wall_thickness=wall_thickness_geometry,
                require_brep=args.require_brep,
                metadata={
                    "wall_thickness_m": args.wall_thickness,
                    "material": mat.name if mat else None,
                    "hardware_qualified": False,
                },
            )
        except RuntimeError as exc:
            print(red(f"\n    CAD export failed: {exc}"))
            return 2
        representation = step_representation(step_path)
        summary["wall_geometry"]["step"] = representation
        artifacts.append(step_path.name)
        badge = green("true B-rep") if representation == "brep" else yellow("faceted fallback")
        print(f"\n    STEP solid: {badge}  →  {step_path.name}")

        if sized_wall_profile is not None:
            try:
                jacket_step = export_step(
                    jacket_inner_x, jacket_inner_r, args.out / "jacket.step",
                    n_angular=96,
                    wall_thickness=sized_wall_profile.t_jacket,
                    require_brep=args.require_brep,
                    metadata={
                        "role": "regen_closeout_jacket",
                        "material": args.jacket_material or (mat.name if mat else None),
                        "hardware_qualified": False,
                    },
                )
            except RuntimeError as exc:
                print(red(f"\n    Jacket CAD export failed: {exc}"))
                return 2
            jacket_representation = step_representation(jacket_step)
            summary["wall_geometry"]["jacket_step"] = jacket_representation
            artifacts.append(jacket_step.name)

        if args.regen_brep:
            try:
                from raosim.regen_cad import (
                    cadquery_available,
                    export_channel_wall_step,
                )
                if not cadquery_available():
                    raise RuntimeError(
                        "CadQuery/OpenCascade is required by --regen-brep"
                    )
                _ncad = int(wall_profile.channel_count)
                network = " + plenums/ports" if args.regen_manifolds else ""
                print(dim(
                    f"    full-N channel-wall B-rep ({_ncad} patterned ribs"
                    f"{network}; one fused solid)…"
                ))
                regen_step_path = args.out / "regen.step"
                rb = export_channel_wall_step(
                    wall_profile,
                    regen_step_path,
                    max_sections=args.regen_cad_sections,
                    include_manifolds=args.regen_manifolds,
                    ports_per_manifold=args.regen_ports_per_manifold,
                    port_area_ratio=args.regen_port_area_ratio,
                    port_diameter=args.regen_port_diameter,
                )
                artifacts.append(regen_step_path.name)
                summary["wall_geometry"]["regen_step"] = {
                    "representation": rb["representation"],
                    "single_solid": rb["single_solid"],
                    "solid_count": rb["solid_count"],
                    "channel_count": rb["channel_count"],
                    "void_fraction": rb["void_fraction"],
                    "include_manifolds": rb["include_manifolds"],
                    "manifold_metrics": rb["manifold_metrics"],
                    "model": rb["model"],
                }
                print(green(
                    f"    manufacturing solid: {rb['channel_count']} full-N "
                    f"channel gaps, one body, void {rb['void_fraction']*100:.1f}%"
                    f"  →  {regen_step_path.name}"
                ))
            except Exception as exc:
                print(red(f"\n    Cooling-aware CAD export failed: {exc}"))
                return 2

    if args.cad in ("ipt", "both") and step_path is not None:
        authoritative_step = regen_step_path or step_path
        manifest = package_ipt_request(
            authoritative_step, args.out / "wall_ipt_manifest.json",
            metadata={
                "wall_thickness_m": args.wall_thickness,
                "material": mat.name if mat else None,
                "authoritative_step": str(authoritative_step),
                "companion_jacket_step": (
                    str(args.out / "jacket.step")
                    if sized_wall_profile is not None else None
                ),
            },
        )
        artifacts.append(manifest.name)
        summary["wall_geometry"]["ipt"] = "inventor_import_manifest"

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
            wall_profile=wall_profile,
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
