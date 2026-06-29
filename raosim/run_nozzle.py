"""
run_nozzle.py — the main runner: solve the contour with the
differentiable / MOC backend, then (optionally) add the regen cooling
coils, run the cooling analysis, and export everything.

Pipeline
--------
1. Solve the Rao variational / MOC nozzle boundary-value problem
   (``solve_rao_bvp``) on the chosen backend — ``--backend jax`` is the
   differentiable Optimistix-LM path with exact autodiff Jacobians;
   ``--backend numpy`` is the SciPy finite-difference oracle.  The wall
   is built by NASA's BDE-region method-of-characteristics march
   (``wall_method="bde"``), i.e. the real characteristic contour, not
   the chart Bézier.
2. Build the chamber from ``L*`` and contraction ratio using the same
   throat geometry as the nozzle.
3. ``--regen`` adds the N regenerative cooling channels (axial, or
   helical "coils" with ``--helix-turns``) wrapping the solved wall.
4. ``--thermal`` runs the coupled Bartz + Sieder-Tate cooling analysis
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

    # seed diagnostic (no LM solve). Invalid seed geometry is reported and
    # deliberately not exported:
    PYTHONPATH=. python scripts/run_nozzle.py --max-nfev 0 --regen \
        --out builds/run_smoke

Note: the full ``--backend jax`` solve (max_nfev ~4000 + weight ladder)
runs for minutes — it is a host job. ``--max-nfev 0`` evaluates the seed
only and is useful for diagnostics; hard geometry-gate failures block export.
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
        ("Chamber", [
            ("--l-star", "characteristic length Vc/At [m]"),
            ("--contraction-ratio", "chamber/throat area ratio Ac/At"),
            ("--shoulder-radius-factor", "chamber shoulder radius / Rt"),
            ("--minimum-cylindrical-length", "minimum useful cylinder [m]"),
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
            ("--wall-thickness", "uniform reference thickness [m]"),
            ("--wall-sizing {scalar,regen}", "scalar wall input or regen+SP-125 sizing"),
            ("--helix-turns", "0 = axial, >0 = helical coils"),
            ("--coolant / --coolant-mdot", "rp1/methane/lh2/water/ethanol; flow [kg/s]"),
            ("--coolant-inlet-temperature", "coolant inlet [K]; cryogenic defaults by fluid"),
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
            ("--t-hot-min / --t-hot-max", "process floor / search ceiling for liner [m]"),
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
    inspect_stl,
    package_ipt_request,
    step_representation,
)
from raosim.chamber_geometry import (
    chamber_contour,
    failed_thrust_chamber_geometry_checks,
    full_engine_contour,
)
from raosim.materials import get_material, material_names, material_table
from raosim.physics import default_coolant_inlet_temperature
from raosim.rao_variational import RaoSolverConfig
from raosim.regen_geometry import generate_regen_nozzle
from raosim.throat_geometry import ThroatGeometrySpec


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


def _default_coolant_inlet_temperature(coolant: str) -> float:
    """Backward-compatible alias for the central physics-layer resolver."""
    return default_coolant_inlet_temperature(coolant)


def _apply_wall_sizing_mode(args, parser, argv) -> None:
    """Resolve the user-facing wall-sizing mode into the existing switches."""
    wall_thickness_given = "--wall-thickness" in argv
    if args.wall_sizing == "scalar":
        if args.size_wall:
            parser.error("--wall-sizing scalar cannot be combined with --size-wall")
        args._wall_sizing_mode = "scalar"
        args._wall_thickness_source = (
            "user_supplied_scalar" if wall_thickness_given else "default_scalar"
        )
        return

    if args.wall_sizing == "regen":
        args.regen = True
        args.size_wall = True
        args._wall_sizing_mode = "regen_thermostructural"
        args._wall_thickness_source = (
            "user_supplied_seed_for_regen_sizing"
            if wall_thickness_given else "default_seed_for_regen_sizing"
        )
        return

    if args.size_wall:
        args.regen = True
        args._wall_sizing_mode = "regen_thermostructural"
        args._wall_thickness_source = (
            "user_supplied_seed_for_regen_sizing"
            if wall_thickness_given else "default_seed_for_regen_sizing"
        )
    else:
        args._wall_sizing_mode = "scalar"
        args._wall_thickness_source = (
            "user_supplied" if wall_thickness_given
            else "default_uniform_reference"
        )


def _interactive(args) -> None:
    """Prompt for the design dimensions + options, overriding ``args``."""
    print(dim("Interactive setup — press Enter to accept each [default].  Ctrl-C to abort."))
    _section("Nozzle dimensions")
    args.rt = _prompt("Throat radius Rt [m]", args.rt)
    args.epsilon = _prompt("Expansion ratio  epsilon = Ae/At", args.epsilon)
    args.length_pct = _prompt("Bell length [% of a 15-deg cone]", args.length_pct)
    args.gamma = _prompt("Specific-heat ratio gamma", args.gamma)
    args.pa_over_p0 = _prompt("Ambient/chamber pressure ratio pa/p0", args.pa_over_p0)
    _section("Chamber dimensions")
    args.l_star = _prompt("Characteristic length L* [m]", args.l_star)
    args.contraction_ratio = _prompt(
        "Contraction ratio Ac/At", args.contraction_ratio
    )
    shoulder_default = (
        args.shoulder_radius_factor
        if args.shoulder_radius_factor is not None else 0.25
    )
    args.shoulder_radius_factor = _prompt(
        "Shoulder radius / Rt", shoulder_default
    )
    cylinder_default = (
        args.minimum_cylindrical_length
        if args.minimum_cylindrical_length is not None else 1e-6
    )
    args.minimum_cylindrical_length = _prompt(
        "Minimum cylindrical length [m]", cylinder_default
    )
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
        sizing_default = args.wall_sizing or (
            "regen" if args.size_wall else "scalar"
        )
        sizing_mode = _prompt(
            "Wall sizing mode (scalar / regen)", sizing_default, str
        ).strip().lower()
        if sizing_mode in {"regen", "regen+sizing", "sized", "size-wall"}:
            args.wall_sizing = "regen"
            args.size_wall = True
        else:
            args.wall_sizing = "scalar"
            args.size_wall = False
            args.wall_thickness = _prompt(
                "Uniform hot-wall thickness [m]", args.wall_thickness
            )
        args.channel_height = _prompt("Channel height/depth [m]", args.channel_height)
        args.helix_turns = _prompt("Helix turns (0 = axial, >0 = helical coils)",
                                   args.helix_turns)
        args.auto_size = _prompt_bool(
            "Auto-size channel count & width from the cooling requirement?",
            bool(args.auto_size))
        if args.auto_size or args.size_wall:
            print(dim("    (you give the requirements; the coolant flow comes "
                      "from the engine cycle, and geometry is solved for)"))
            args.pc = _prompt("Chamber pressure Pc [Pa]", args.pc)
            args.coolant = _prompt("Coolant (rp1/methane/lh2/water/ethanol)",
                                   args.coolant, str)
            inlet_default = (
                args.coolant_inlet_temperature
                if args.coolant_inlet_temperature is not None
                else _default_coolant_inlet_temperature(args.coolant)
            )
            args.coolant_inlet_temperature = _prompt(
                "Coolant inlet temperature [K]", inlet_default
            )
            args.mixture_ratio = _prompt("Mixture ratio O/F (sets coolant flow)",
                                         args.mixture_ratio)
            args.margin_target = _prompt("Required cooling margin (limit/peak ≥)",
                                         args.margin_target)
            args.wall_temp_limit = _prompt("Max wall temperature [K]", args.wall_temp_limit)
            args.dp_budget = _prompt("Pressure-drop budget [bar]", args.dp_budget)
            if args.size_wall:
                args.structural_fos = _prompt(
                    "Structural factor of safety", args.structural_fos
                )
                args.required_cycles = _prompt(
                    "Required thermal cycles", args.required_cycles
                )
                args.t_hot_min = _prompt(
                    "Minimum hot-wall thickness [m]", args.t_hot_min
                )
                args.t_hot_max = _prompt(
                    "Maximum hot-wall thickness [m]", args.t_hot_max
                )
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
                inlet_default = (
                    args.coolant_inlet_temperature
                    if args.coolant_inlet_temperature is not None
                    else _default_coolant_inlet_temperature(args.coolant)
                )
                args.coolant_inlet_temperature = _prompt(
                    "Coolant inlet temperature [K]", inlet_default
                )
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
        throat_upstream_radius_factor=args.ru_factor,
        throat_downstream_radius_factor=args.rd_factor,
        kernel_d_fraction_max=0.7,
        thetaN_guess_deg=args.theta_b_guess,
    )
    return rv.solve_rao_bvp(cfg)


def _print_injector_panel(inj) -> None:
    """Console panel for a sized pintle injector."""
    fs = inj.streams["fuel"]
    os = inj.streams["oxidizer"]
    print(f"    {green('streams:'):<16}radial={inj.radial_stream} "
          f"({'slots' if inj.radial_stream == inj.slots.role else 'annulus'})  "
          f"Dp={inj.pintle_diameter*1e3:.1f} mm  "
          f"feed: fuel {inj.feed['fuel'].name} / ox {inj.feed['oxidizer'].name}")
    for label, s in (("fuel", fs), ("ox", os)):
        we = "—" if (s.weber != s.weber) else f"{s.weber:.0f}"
        oh = "—" if (s.ohnesorge != s.ohnesorge) else f"{s.ohnesorge:.3f}"
        print(f"    {green(label + ':'):<16}{s.geometry:7s} "
              f"ṁ={s.mdot:.3f} kg/s  ΔP={s.dp/1e5:.1f} bar  "
              f"A={s.area*1e6:.2f} mm²  v={s.velocity:.0f} m/s  "
              f"Re={s.reynolds:.0f}  We={we}  Oh={oh}")
    sd, ad = inj.slots.detail, inj.annulus.detail
    print(f"    {green('geometry:'):<16}annulus gap={ad['gap']*1e3:.3f} mm   "
          f"slot {sd['slot_width']*1e3:.3f}×{sd['slot_height']*1e3:.3f} mm "
          f"×{inj.slot_count}   web={inj.minimum_web*1e3:.3f} mm   "
          f"blockage={inj.blockage_factor:.2f}")
    wall = (f"{inj.spray_wall_axial_distance*1e3:.0f} mm"
            if inj.spray_wall_axial_distance == inj.spray_wall_axial_distance
            and inj.spray_wall_axial_distance != float("inf") else "—")
    print(f"    {green('spray:'):<16}TMR={inj.total_momentum_ratio:.2f}   "
          f"half-angle={inj.spray_half_angle_deg:.0f}°   "
          f"wall@{wall}   slot/gap={inj.slot_to_annulus_width_ratio:.2f}")
    at = getattr(inj, "atomization", None)
    if at is not None:
        lim = at.streams[at.limiting_role]
        print(f"    {green('atomization:'):<16}"
              f"SMD {lim.sauter_mean_diameter*1e6:.0f} µm ({at.limiting_role})   "
              f"L_comb {at.combustion_length*1e3:.0f} mm   "
              f"margin {at.development_margin:.2f}   "
              f"η_c*≈{at.predicted_cstar_efficiency:.2f} "
              + dim("(vaporization-limited surrogate)"))
    th = getattr(inj, "thermal", None)
    if th is not None:
        twg = (th.tip_wall_temperature if th.limiting == "tip"
               else th.face_wall_temperature)
        print(f"    {green('face/tip:'):<16}"
              f"{th.limiting} T_wg≈{twg:.0f} K vs {th.wall_temperature_limit:.0f} K   "
              f"margin {th.governing_margin:.2f} "
              + dim("(recirculation screen)"))
    st = getattr(inj, "stability", None)
    if st is not None:
        print(f"    {green('stability:'):<16}"
              f"chug {st.chug_status}   "
              f"L1 {st.f_L1:.0f} Hz  T1 {st.f_T1:.0f} Hz   "
              f"τ·f_L1={st.reduced_frequency_L1:.2f} "
              + dim("(n-τ screen)"))
    n_pass = sum(g.status == "pass" for g in inj.gates)
    n_warn = sum(g.status == "warn" for g in inj.gates)
    n_fail = sum(g.status == "fail" for g in inj.gates)
    print(f"    {green('gates:'):<16}{green(str(n_pass)+' pass')}  "
          f"{yellow(str(n_warn)+' warn')}  "
          f"{red(str(n_fail)+' fail') if n_fail else dim('0 fail')}")
    for g in inj.gates:
        if g.status == "fail":
            print(red(f"      ✗ {g.name}: {g.detail}"))
        elif g.status == "warn":
            print(yellow(f"      ● {g.name}: {g.detail}"))
    verdict = (green("✓ feasible (no failing gates)") if inj.feasible
               else red("✗ infeasible — failing gates above"))
    print(f"    {green('verdict:'):<16}{verdict}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    # nozzle
    ap.add_argument("--rt", type=float, default=0.020, help="throat radius [m]")
    ap.add_argument("--target-thrust", type=float, default=None,
                    help="design thrust [N]; sizes Rt from F/(Cf·Pc). "
                         "Mutually exclusive with an explicit --rt.")
    ap.add_argument("--epsilon", type=float, default=10.0)
    ap.add_argument("--length-pct", type=float, default=80.0)
    ap.add_argument("--gamma", type=float, default=1.24,
                    help="combustion-product gamma; overridden by the resolved "
                         "propellant unless passed explicitly")
    ap.add_argument("--pa-over-p0", type=float, default=0.01,
                    help="design ambient/chamber pressure ratio (sets Pa)")
    # propellant + thermochemistry (drives gamma, c*, Tc, mass flow)
    ap.add_argument("--thermo-mode", choices=("cea", "constant-gamma"),
                    default="constant-gamma",
                    help="cea = RocketCEA when installed (validated); "
                         "constant-gamma = built-in literature table (screening)")
    ap.add_argument("--propellant", default=None,
                    help="named combustion pair, e.g. 'LOX/RP-1' "
                         "(see the built-in table); or use --oxidizer/--fuel")
    ap.add_argument("--oxidizer", default=None, help="oxidizer name")
    ap.add_argument("--fuel", default=None, help="fuel name")
    ap.add_argument("--oxidizer-inlet-temperature", type=float, default=None,
                    help="oxidizer feed temperature [K]")
    ap.add_argument("--fuel-inlet-temperature", type=float, default=None,
                    help="fuel feed temperature [K]")
    ap.add_argument("--oxidizer-inlet-pressure", type=float, default=None,
                    help="oxidizer feed (manifold-inlet) pressure [Pa]")
    ap.add_argument("--fuel-inlet-pressure", type=float, default=None,
                    help="fuel feed (manifold-inlet) pressure [Pa]")
    ap.add_argument("--oxidizer-phase", choices=("auto", "liquid", "gas"),
                    default="auto", help="oxidizer injection phase")
    ap.add_argument("--fuel-phase", choices=("auto", "liquid", "gas"),
                    default="auto", help="fuel injection phase")
    ap.add_argument("--eta-cstar", type=float, default=None,
                    help="combustion (c*) efficiency override")
    ap.add_argument("--eta-cf", type=float, default=None,
                    help="nozzle (thrust-coefficient) efficiency override")
    # chamber and shared throat
    ap.add_argument("--l-star", type=float, default=1.0,
                    help="chamber characteristic length Vc/At [m]")
    ap.add_argument("--contraction-ratio", type=float, default=2.5,
                    help="chamber/throat area ratio Ac/At")
    ap.add_argument("--shoulder-radius-factor", type=float, default=None,
                    help="chamber shoulder radius / Rt; geometric placeholder "
                         "0.25 when omitted")
    ap.add_argument("--minimum-cylindrical-length", type=float, default=None,
                    help="minimum cylindrical chamber length [m]; geometric "
                         "floor 1e-6 when omitted")
    ap.add_argument("--convergent-angle", type=float, default=45.0,
                    help="shared chamber/nozzle convergent half-angle [deg]")
    ap.add_argument("--ru-factor", type=float, default=1.5,
                    help="shared upstream throat radius / Rt")
    ap.add_argument("--rd-factor", type=float, default=0.382,
                    help="shared downstream throat radius / Rt")
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
    ap.add_argument("--channel-roughness", type=float, default=0.0,
                    help="mean internal channel roughness [m]; zero = ideal smooth")
    ap.add_argument("--gate-coolant-chemistry", action="store_true",
                    help="make coolant-specific chemistry limits (for example "
                         "the RP-1 coking screen) gate auto-sizing feasibility")
    ap.add_argument("--curvature-correction", action="store_true",
                    help="opt into the Niino-Kumakawa/Taylor curved-channel "
                         "heat-transfer screen")
    ap.add_argument("--channel-height-min", type=float, default=None,
                    help="minimum channel depth searched by --size-wall [m]")
    ap.add_argument("--channel-height-max", type=float, default=None,
                    help="maximum channel depth searched by --size-wall [m]")
    ap.add_argument("--channel-height-steps", type=int, default=3,
                    help="number of channel-depth candidates for --size-wall")
    ap.add_argument("--wall-thickness", type=float, default=0.001,
                    help="uniform reference thickness [m]; --size-wall "
                         "replaces it with a station-wise analyzed profile")
    ap.add_argument("--wall-sizing", choices=("scalar", "regen"), default=None,
                    help="wall-thickness mode: scalar uses --wall-thickness "
                         "uniformly; regen enables regenerative hot-wall + "
                         "channel sizing using the SP-125 structural screen "
                         "(needs --material)")
    ap.add_argument("--helix-turns", type=float, default=0.0,
                    help="0 = axial channels; >0 = helical coils")
    ap.add_argument("--coolant", default="rp1")
    ap.add_argument("--coolant-mdot", type=float, default=10.0, help="[kg/s]")
    ap.add_argument("--coolant-inlet-temperature", type=float, default=None,
                    help="coolant inlet temperature [K]; defaults to 120 K "
                         "for methane, 25 K for LH2, and 300 K otherwise")
    ap.add_argument("--coolant-outlet-pressure", type=float, default=None,
                    help="absolute coolant pressure at jacket outlet [Pa]; "
                         "default Pc + injector pressure drop")
    ap.add_argument("--injector-pressure-drop", type=float, default=0.0,
                    help="DEPRECATED absolute cooling-outlet loss [Pa]; prefer "
                         "--fuel-injector-dp-fraction. Only a single boundary "
                         "loss, not a complete injector.")
    ap.add_argument("--fuel-injector-dp-fraction", type=float, default=None,
                    help="fuel injector pressure drop as a fraction of Pc "
                         "(ΔP_f/Pc); sets the regen coolant outlet boundary to "
                         "Pc·(1+χ_f) when fuel is the coolant")
    ap.add_argument("--oxidizer-injector-dp-fraction", type=float, default=None,
                    help="oxidizer injector pressure drop as a fraction of Pc "
                         "(ΔP_o/Pc)")
    # injector (pintle) — sized from the cycle mass-flow split
    ap.add_argument("--injector", choices=("none", "pintle"), default="none",
                    help="generate a pintle injector from the operating point")
    ap.add_argument("--injector-sizing", choices=("auto", "fixed"),
                    default="auto",
                    help="auto: derive openings from ṁ/ΔP; fixed: evaluate "
                         "supplied geometry without resizing")
    ap.add_argument("--pintle-radial-stream", choices=("fuel", "oxidizer"),
                    default="fuel", help="which stream is slotted (radial)")
    ap.add_argument("--fuel-discharge-coefficient", type=float, default=0.7,
                    help="fuel metering discharge coefficient Cd")
    ap.add_argument("--oxidizer-discharge-coefficient", type=float, default=0.7,
                    help="oxidizer metering discharge coefficient Cd")
    ap.add_argument("--pintle-diameter", type=float, default=None,
                    help="pintle diameter [m] (annulus + slot anchor); "
                         "default 0.30·chamber diameter")
    ap.add_argument("--pintle-slot-count", type=int, default=24,
                    help="number of radial slots/holes")
    ap.add_argument("--pintle-slot-aspect-ratio", type=float, default=1.0,
                    help="slot height/width for auto-sizing")
    ap.add_argument("--pintle-deflector-angle", type=float, default=0.0,
                    help="radial-stream deflector angle [deg]")
    ap.add_argument("--pintle-target-momentum-ratio", type=float, default=None,
                    help="optional target radial/axial momentum ratio; "
                         "currently gates the achieved design")
    ap.add_argument("--pintle-impingement-distance", type=float, default=None,
                    help="distance from openings to stream interaction [m]")
    ap.add_argument("--injector-min-feature", type=float, default=3.0e-4,
                    help="manufacturing floor for gaps/slots/ligaments [m]")
    ap.add_argument("--allow-infeasible-injector", action="store_true",
                    help="export the chamber even when injector gates fail "
                         "(default: failing gates block export, exit nonzero)")
    ap.add_argument("--throttle-map", default=None,
                    help="movable-sleeve throttle study: comma-separated "
                         "throttle levels in (0,1], e.g. '0.2,0.4,0.6,0.8,1.0'")
    ap.add_argument("--throttle-pc-exponent", type=float, default=1.0,
                    help="Pc(f)=Pc·f^exp for the throttle map (1=Pc∝ṁ, "
                         "0=constant Pc)")
    ap.add_argument("--injector-cad",
                    choices=("none", "auto", "reference", "parts", "step"),
                    default="auto",
                    help="pintle package CAD mode with --injector pintle: "
                         "none writes only JSON/CSV/SVG/PNG; reference writes a "
                         "single CAD-neutral reference file; parts also writes "
                         "named part files; auto uses parts when CadQuery is "
                         "available and otherwise keeps the mandatory package; "
                         "step is the legacy alias for required STEP parts")
    ap.add_argument("--injector-cad-format", choices=("step", "stl", "dxf"),
                    default="step",
                    help="format for --injector-cad reference/parts/auto "
                         "(STEP default; DXF is a 2-D meridional profile)")
    ap.add_argument("--pintle-sleeve", action="store_true",
                    help="include the movable sleeve body in the pintle CAD")
    # fixed-geometry overrides (only used with --injector-sizing fixed)
    ap.add_argument("--pintle-annulus-gap", type=float, default=None)
    ap.add_argument("--pintle-slot-width", type=float, default=None)
    ap.add_argument("--pintle-slot-height", type=float, default=None)
    ap.add_argument("--pintle-slot-depth", type=float, default=None)
    ap.add_argument("--pintle-tip-radius", type=float, default=None)
    ap.add_argument("--pintle-body-length", type=float, default=None)
    ap.add_argument("--injector-face-thickness", type=float, default=None)
    ap.add_argument("--injector-face-od", type=float, default=None)
    ap.add_argument("--flange-od", type=float, default=None,
                    help="optional chamber/injector flange outer diameter [m]")
    ap.add_argument("--flange-length", type=float, default=None,
                    help="optional upstream axial flange length [m]")
    ap.add_argument("--bolt-count", type=int, default=None,
                    help="optional injector/chamber bolt count")
    ap.add_argument("--bolt-circle", type=float, default=None,
                    help="optional bolt circle diameter [m]")
    ap.add_argument("--bolt-hole", type=float, default=None,
                    help="optional bolt hole diameter [m]")
    ap.add_argument("--bolt-diameter", type=float, default=None,
                    help="optional actual bolt/tensile diameter [m]")
    ap.add_argument("--bolt-allowable-stress", type=float, default=None,
                    help="optional bolt allowable tensile stress [Pa]")
    ap.add_argument("--joint-separation-factor", type=float, default=1.5,
                    help="clamp-load factor on Pc*pi*Rc^2 separating load")
    ap.add_argument("--coolant-property-backend",
                    choices=("auto", "constant", "coolprop"), default="auto",
                    help="coolant properties: CoolProp methane/LH2 when "
                         "available, explicit constants, or require CoolProp")
    ap.add_argument("--hydraulic-network", action="store_true",
                    help="solve every channel branch plus annular inlet/outlet "
                         "plenums and discrete ports")
    ap.add_argument("--plenum-area-ratio", type=float, default=2.0,
                    help="plenum cross-section / total channel flow area")
    ap.add_argument("--port-loss-coefficient", type=float, default=1.5,
                    help="minor-loss K for each manifold port")
    ap.add_argument("--radiation-model",
                    choices=("none", "leccese_gray", "spectral"),
                    default="none",
                    help="participating-gas radiation model added to Bartz")
    ap.add_argument("--radiation-family",
                    choices=("methane", "hydrogen"), default=None,
                    help="propellant family for the Leccese gray preset")
    ap.add_argument("--radiation-path-length", type=float, default=None,
                    help="participating-gas path length [m]; default local radius")
    ap.add_argument("--radiation-wall-emissivity", type=float, default=1.0)
    ap.add_argument("--radiation-bands-json", default=None,
                    help="JSON list of spectral bands with name, weight, and "
                         "absorption_coefficient [1/m]")
    ap.add_argument("--boiling-chf", action="store_true",
                    help="evaluate real-fluid phase state and CHF diagnostics")
    ap.add_argument("--gate-chf", action="store_true",
                    help="let the subcritical CHF screen gate feasibility")
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
    ap.add_argument("--mixture-ratio", type=float, default=None,
                    help="oxidiser/fuel ratio (sets the cycle flow split); "
                         "defaults to the selected propellant's nominal O/F")
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
    ap.add_argument("--t-hot-min", type=float, default=0.0005,
                    help="manufacturing/degradation floor for --size-wall [m]; "
                         "must come from the selected construction process")
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
    _apply_wall_sizing_mode(args, ap, sys.argv)
    if args.regen_manifolds:
        args.regen_brep = True
        args.hydraulic_network = True
    if args.regen_brep and not args.regen:
        ap.error("--regen-brep requires --regen")
    if args.regen_brep and args.cad == "none":
        ap.error("--regen-brep requires --cad step, ipt, or both")
    if args.l_star <= 0.0:
        ap.error("--l-star must be positive")
    if args.contraction_ratio <= 1.0:
        ap.error("--contraction-ratio must be greater than one")
    if (args.flange_od is None) != (args.flange_length is None):
        ap.error("--flange-od and --flange-length must be supplied together")
    for _name in (
        "flange_od", "flange_length", "bolt_circle", "bolt_hole",
        "bolt_diameter", "bolt_allowable_stress", "injector_face_od",
        "injector_face_thickness",
    ):
        _value = getattr(args, _name)
        if _value is not None and _value <= 0.0:
            ap.error(f"--{_name.replace('_', '-')} must be positive")
    if args.bolt_count is not None and args.bolt_count <= 0:
        ap.error("--bolt-count must be positive")
    if args.joint_separation_factor <= 0.0:
        ap.error("--joint-separation-factor must be positive")
    if args.shoulder_radius_factor is None:
        args.shoulder_radius_factor = 0.25
        args._shoulder_radius_source = "geometric_placeholder"
    else:
        args._shoulder_radius_source = "user_supplied"
    if args.shoulder_radius_factor <= 0.0:
        ap.error("--shoulder-radius-factor must be positive")
    if args.minimum_cylindrical_length is None:
        args.minimum_cylindrical_length = 1e-6
        args._minimum_cylinder_source = "geometric_placeholder"
    else:
        args._minimum_cylinder_source = "user_supplied"
    if args.minimum_cylindrical_length <= 0.0:
        ap.error("--minimum-cylindrical-length must be positive")
    if args.wall_thickness <= 0.0:
        ap.error("--wall-thickness must be positive")
    if not (0.0 < args.t_hot_min <= args.t_hot_max):
        ap.error("--t-hot-min and --t-hot-max must satisfy 0 < min <= max")
    try:
        radiation_bands = (
            tuple(json.loads(args.radiation_bands_json))
            if args.radiation_bands_json else ()
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        ap.error(f"--radiation-bands-json is invalid: {exc}")
    args._cooling_options = {
        "coolant_property_backend": args.coolant_property_backend,
        "hydraulic_network": args.hydraulic_network,
        "ports_per_manifold": args.regen_ports_per_manifold,
        "port_area_ratio": args.regen_port_area_ratio,
        "port_diameter": args.regen_port_diameter,
        "plenum_area_ratio": args.plenum_area_ratio,
        "port_loss_coefficient": args.port_loss_coefficient,
        "radiation_model": args.radiation_model,
        "radiation_propellant_family": args.radiation_family,
        "radiation_path_length": args.radiation_path_length,
        "radiation_wall_emissivity": args.radiation_wall_emissivity,
        "radiation_bands": radiation_bands,
        "boiling_chf": args.boiling_chf,
        "gate_chf": args.gate_chf,
    }
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
        _apply_wall_sizing_mode(args, ap, sys.argv)

    if args.coolant_inlet_temperature is None:
        args.coolant_inlet_temperature = _default_coolant_inlet_temperature(
            args.coolant
        )
        args._coolant_inlet_temperature_source = "central_coolant_default"
    else:
        args._coolant_inlet_temperature_source = "user_supplied"
    if args.coolant_inlet_temperature <= 0.0:
        ap.error("--coolant-inlet-temperature must be positive")
    args._cooling_options["coolant_inlet_temperature"] = (
        args.coolant_inlet_temperature
    )

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
    if args.size_wall and mat is None:
        print(red(
            "    --size-wall needs --material for the structural properties "
            "(E/α/ν/yield); see --list-materials."
        ))
        return 2

    # ---- resolve propellant + operating point (F → Rt → ṁ → ṁ_f, ṁ_o) ---
    # The cooling cycle AND the injector both need a real propellant and the
    # closed mass-flow split, not a bare gamma.  Resolve thermochemistry once
    # here so the contour, separation, cooling, and injector share it.
    from raosim.cea import (
        THERMO_CEA_FROZEN, THERMO_CONSTANT_GAMMA, resolve_thermochemistry,
    )
    from raosim.engine import compute_engine_performance
    from raosim.design import throat_radius_for_target_thrust
    from raosim.propellants import custom_propellant

    rt_explicit = "--rt" in sys.argv
    gamma_explicit = "--gamma" in sys.argv
    if args.target_thrust is not None and rt_explicit:
        ap.error("--target-thrust and --rt are mutually exclusive")
    if args.target_thrust is not None and args.target_thrust <= 0.0:
        ap.error("--target-thrust must be positive")

    prop_name = args.propellant
    if prop_name is None and args.oxidizer and args.fuel:
        prop_name = f"{args.oxidizer}/{args.fuel}"
    # Individual oxidizer/fuel identities for the injector feed states.
    ox_name, fuel_name = args.oxidizer, args.fuel
    if (ox_name is None or fuel_name is None) and args.propellant and \
            "/" in args.propellant:
        ox_name, fuel_name = (s.strip() for s in args.propellant.split("/", 1))
    args._ox_name = ox_name
    args._fuel_name = fuel_name
    thermo_mode = (THERMO_CEA_FROZEN if args.thermo_mode == "cea"
                   else THERMO_CONSTANT_GAMMA)
    prop_warnings: list[str] = []

    # Default the mixture ratio from the selected propellant's nominal O/F
    # (so e.g. LOX/LH2 splits flow near its own O/F, not a hard-coded 2.6).
    # Must happen before thermochemistry — CEA needs the mixture ratio.
    if args.mixture_ratio is None:
        nominal_of = None
        if prop_name is not None:
            try:
                from raosim.propellants import get_propellant
                nominal_of = get_propellant(prop_name).OF
            except Exception:
                nominal_of = None
        if nominal_of and nominal_of > 0:
            args.mixture_ratio = float(nominal_of)
            prop_warnings.append(
                f"--mixture-ratio not given; defaulted to the propellant "
                f"nominal O/F = {args.mixture_ratio:g}.")
        else:
            args.mixture_ratio = 2.6
            prop_warnings.append(
                "--mixture-ratio not given and no propellant O/F available; "
                "defaulted to 2.6.")

    if prop_name is None:
        # Back-compat: no propellant named → an unspecified custom gas from
        # gamma alone (Mw/Tc are placeholders).  Mass flow still closes, but
        # the absolute numbers are only as good as the guess.
        prop = custom_propellant(
            gamma=args.gamma, Mw=0.022, Tc=3500.0,
            eta_cstar=args.eta_cstar, eta_CF=args.eta_cf,
            source="unspecified custom gas (gamma only; Mw=22 g/mol, "
                   "Tc=3500 K placeholders) — name a propellant for real "
                   "thermochemistry",
        )
        prop_warnings.append(
            "No --propellant / --oxidizer+--fuel given; using a placeholder "
            "custom gas (gamma only). Performance and mass-flow numbers are "
            "indicative only."
        )
    else:
        try:
            thermo = resolve_thermochemistry(
                thermo_mode=thermo_mode, propellant_name=prop_name,
                Pc=args.pc, mixture_ratio=args.mixture_ratio,
                oxidizer=args.oxidizer, fuel=args.fuel,
                eta_Isp=0.95, epsilon=args.epsilon, require_cea=False,
            )
        except Exception as exc:
            ap.error(f"could not resolve propellant '{prop_name}': {exc}")
        prop = thermo.propellant
        prop_warnings.extend(thermo.warnings)
        # CLI eta overrides win over the table/CEA split.
        if args.eta_cstar is not None or args.eta_cf is not None:
            name = prop.name
            prop = custom_propellant(
                gamma=prop.gamma, Mw=prop.Mw, Tc=prop.Tc, OF=prop.OF,
                eta_cstar=(args.eta_cstar if args.eta_cstar is not None
                           else prop.eta_cstar),
                eta_CF=(args.eta_cf if args.eta_cf is not None
                        else prop.eta_CF),
                source=prop.source,
            )
            prop.name = name
    if not gamma_explicit:
        args.gamma = prop.gamma
    args._prop = prop
    args._prop_warnings = prop_warnings

    Pa = args.pa_over_p0 * args.pc
    if args.target_thrust is not None:
        args.rt = throat_radius_for_target_thrust(
            args.target_thrust, args.pc, Pa, args.epsilon, prop)
    args._performance = compute_engine_performance(
        Pc=args.pc, Pa=Pa, Rt=args.rt, epsilon=args.epsilon, prop=prop)
    perf = args._performance
    args._mdot = perf.m_dot
    args._mdot_f = perf.m_dot / (1.0 + max(args.mixture_ratio, 0.0))
    args._mdot_o = args.mixture_ratio * args._mdot_f
    args._design_ambient = Pa

    # Injector pressure-drop fractions (replace the single legacy loss).
    # Explicit flag wins; else derive from the legacy absolute loss; else use
    # a standard ~20% Pc injector stiffness so --injector works out of the box.
    if args.fuel_injector_dp_fraction is None:
        if args.injector_pressure_drop > 0 and args.pc > 0:
            args.fuel_injector_dp_fraction = args.injector_pressure_drop / args.pc
        else:
            args.fuel_injector_dp_fraction = 0.2
    if args.oxidizer_injector_dp_fraction is None:
        args.oxidizer_injector_dp_fraction = args.fuel_injector_dp_fraction
    # The fuel injector drop sets the regen coolant outlet boundary Pc·(1+χ_f)
    # (continuous pressure accounting: jacket -> fuel manifold -> injector).
    # Only override the cooling boundary when the fuel drop is meaningful for
    # this run (explicit flag, or an active pintle) so plain cooling runs keep
    # their existing Pc boundary.
    if "--fuel-injector-dp-fraction" in sys.argv or args.injector == "pintle":
        args.injector_pressure_drop = args.fuel_injector_dp_fraction * args.pc

    args.out.mkdir(parents=True, exist_ok=True)

    # ---- config summary --------------------------------------------------
    print(cyan("▸ " + bold("Build plan")))
    style = (f"helical {args.helix_turns:g} turns" if args.helix_turns
             else "axial") if args.regen else "—"
    wall_plan = (
        f"regen thermostructural sizing (thermal margin≥{args.margin_target:g}, "
        f"SP-125 FoS≥{args.structural_fos:g}, Δp≤{args.dp_budget:g} bar)"
        if args.size_wall else
        f"scalar uniform input ({args.wall_thickness*1e3:g} mm)"
    )
    for k, v in [
        ("nozzle", f"Rt={args.rt*1e3:g} mm, eps={args.epsilon:g}, "
                   f"L={args.length_pct:g}%, gamma={args.gamma:g}"),
        ("chamber", f"L*={args.l_star:g} m, CR={args.contraction_ratio:g}, "
                    f"shoulder={args.shoulder_radius_factor:g} Rt, "
                    f"Lc,min={args.minimum_cylindrical_length:g} m"),
        ("solver", f"{args.backend}  (max_nfev={args.max_nfev}, "
                   f"n_control={args.n_control}, n_kernel={args.n_kernel})"),
        ("regen", (
            (f"wall+channel sizing from requirement "
             f"(margin≥{args.margin_target:g}, FoS≥{args.structural_fos:g}, "
             f"Δp≤{args.dp_budget:g} bar, T_wall≤{args.wall_temp_limit:g} K)"
             if args.size_wall else
             f"auto-size from requirement (margin≥{args.margin_target:g}, "
             f"Δp≤{args.dp_budget:g} bar, T_wall≤{args.wall_temp_limit:g} K)"
             if args.auto_size else
             f"{args.channels} channels, {style}, "
             f"{args.channel_width*1e3:g}×{args.channel_height*1e3:g} mm")
            if args.regen else dim("off"))),
        ("wall sizing", wall_plan),
        ("thermal", (f"on  (Pc={args.pc/1e5:g} bar, {args.coolant} "
                     f"@ {args.coolant_inlet_temperature:g} K, "
                     + (f"MR={args.mixture_ratio:g} → cycle flow"
                        if (args.auto_size or args.size_wall)
                        else f"{args.coolant_mdot:g} kg/s") + ")"
                     if (args.thermal or args.auto_size or args.size_wall) and args.regen
                     else dim("off"))),
        ("material", (f"{mat.name}  (k={args.wall_k:g} W/mK, "
                      f"T≤{args.wall_temp_limit:g} K, "
                      f"{mat.category.replace('_', ' ')})" if mat else
                      dim(f"unspecified  (k={args.wall_k:g} W/mK, "
                          f"T≤{args.wall_temp_limit:g} K)"))),
        ("output", str(args.out)),
    ]:
        print(f"    {green(k+':'):<14}{v}")

    # ---- performance summary  (F, Cf, c*, Isp, ṁ split) -----------------
    print("\n" + cyan("▸ " + bold("Performance")) + dim(f"  ({perf.propellant_name})"))
    thrust_note = (dim("  (Rt from target thrust)") if args.target_thrust
                   else "")
    print(f"    {green('thrust:'):<16}{bold('%.0f N' % perf.thrust)} "
          f"({perf.thrust/1e3:.2f} kN){thrust_note}   "
          f"Isp={bold('%.1f s' % perf.Isp)}   Cf={perf.Cf_actual:.3f}")
    print(f"    {green('c*:'):<16}{perf.c_star:.0f} m/s "
          f"(delivered {perf.c_star_effective:.0f})   "
          f"η_c*={perf.eta_cstar:.3f}  η_CF={perf.eta_CF:.3f}  "
          f"η_Isp={perf.eta_Isp:.3f}")
    print(f"    {green('mass flow:'):<16}{bold('%.3f kg/s' % perf.m_dot)} total   "
          f"ṁ_fuel={args._mdot_f:.3f}  ṁ_ox={args._mdot_o:.3f}  "
          f"(O/F={args.mixture_ratio:g})")
    print(f"    {green('throat/exit:'):<16}Rt={args.rt*1e3:.2f} mm   "
          f"Pa={Pa/1e3:.1f} kPa   Me={perf.Me:.2f}   Pe={perf.Pe/1e3:.1f} kPa")
    for _w in prop_warnings:
        print(yellow(f"    note: {_w}"))
    if prop.source:
        print(dim(f"    source: {prop.source}"))

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

    throat_geometry = ThroatGeometrySpec(
        upstream_radius_ratio=args.ru_factor,
        downstream_radius_ratio=args.rd_factor,
        convergent_half_angle_deg=args.convergent_angle,
    )
    nozzle = sol.to_contour_dict(
        Rt=args.rt, epsilon=args.epsilon, length_pct=args.length_pct,
        pa_over_p0=args.pa_over_p0,
        Ru_factor=args.ru_factor,
        Rd_factor=args.rd_factor,
        convergent_half_angle_deg=args.convergent_angle,
    )
    nozzle["throat_geometry"] = throat_geometry.to_dict()
    nozzle["throat_location"] = throat_geometry.throat_location
    try:
        chamber = chamber_contour(
            args.rt,
            L_star=args.l_star,
            contraction_ratio=args.contraction_ratio,
            throat_geometry=throat_geometry,
            shoulder_radius_factor=args.shoulder_radius_factor,
            minimum_cylindrical_length=args.minimum_cylindrical_length,
        )
        contour = full_engine_contour(chamber, nozzle)
    except ValueError as exc:
        print(red(f"    chamber geometry rejected: {exc}"))
        return 2
    failed_geometry = failed_thrust_chamber_geometry_checks(
        contour["geometry_checks"]
    )
    if failed_geometry:
        checks = contour["geometry_checks"]
        print(red(
            "    thrust-chamber geometry gates failed; no contour or CAD "
            "artifacts were exported."
        ))
        for name in failed_geometry:
            detail = checks.get(name)
            if name == "slope_continuity":
                detail = (
                    f"{checks['maximum_join_angle_deg']:.6g} deg maximum join"
                )
            print(red(f"      - {name}: {detail}"))
        return 2
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
        "chamber": {
            "L_star_m": args.l_star,
            "contraction_ratio": args.contraction_ratio,
            "shoulder_radius_factor": args.shoulder_radius_factor,
            "shoulder_radius_source": args._shoulder_radius_source,
            "minimum_cylindrical_length_m":
                args.minimum_cylindrical_length,
            "minimum_cylindrical_length_source":
                args._minimum_cylinder_source,
            "cylindrical_length_m": chamber["Lc"],
            "target_volume_m3": chamber["V_target"],
            "polyline_frustum_volume_m3": chamber["V_chamber"],
            "geometry_checks": contour["geometry_checks"],
        },
    }
    summary["performance"] = {
        "propellant": perf.propellant_name,
        "thermo_mode": args.thermo_mode,
        "source": prop.source,
        "Pc_pa": args.pc,
        "Pa_pa": Pa,
        "mixture_ratio": args.mixture_ratio,
        "gamma": prop.gamma,
        "Mw_kg_per_mol": prop.Mw,
        "Tc_K": prop.Tc,
        "c_star_ideal_m_s": perf.c_star,
        "c_star_effective_m_s": perf.c_star_effective,
        "eta_cstar": perf.eta_cstar,
        "eta_CF": perf.eta_CF,
        "eta_Isp": perf.eta_Isp,
        "thrust_N": perf.thrust,
        "Isp_s": perf.Isp,
        "Cf_actual": perf.Cf_actual,
        "Me": perf.Me,
        "Pe_pa": perf.Pe,
        "mdot_total_kg_s": perf.m_dot,
        "mdot_fuel_kg_s": args._mdot_f,
        "mdot_oxidizer_kg_s": args._mdot_o,
        "rt_from_target_thrust": args.target_thrust is not None,
        "target_thrust_N": args.target_thrust,
        "warnings": prop_warnings,
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
    axp.set_title(f"Full thrust chamber  eps={args.epsilon:g} L{args.length_pct:g}%  "
                  f"Cf={sol.thrust_coefficient:.3f}")
    fig.tight_layout(); fig.savefig(args.out / "profile.png", dpi=150); fig.clf()
    artifacts = ["contour.csv", "profile.png"]

    # The pintle injector is sized later (after the cooling analysis) so the
    # fuel feed state can come from the regen jacket outlet.  None until then.
    cooling_result = None

    # ---- flow-separation story on the contour (over-expansion) --------
    # Where the wall flow detaches at sea level vs altitude (Schmucker);
    # every nozzle has this story and contour+Pc+gamma are always known.
    try:
        from raosim.plotting import plot_separation_on_contour
        _sep_prop = args._prop
        fig = plot_separation_on_contour(
            contour, args.pc, _sep_prop,
            save_path=args.out / "separation_contour.png", show=show)
        fig.clf()
        artifacts.append("separation_contour.png")
        print(green("    wrote separation_contour.png"))
    except Exception as exc:  # plotting must never break the run
        print(yellow(f"    separation_contour.png skipped: {exc}"))

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
        from raosim.thermal_design import joint_wall_channel_design
        prop = args._prop
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
            channel_roughness=args.channel_roughness,
            gate_coolant_chemistry=args.gate_coolant_chemistry,
            curvature_correction=args.curvature_correction,
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
            t_hot_min=args.t_hot_min, t_hot_max=args.t_hot_max, objective=obj,
            coolant_outlet_pressure=args.coolant_outlet_pressure,
            injector_pressure_drop=args.injector_pressure_drop,
            cooling_options=args._cooling_options)
        print(f"    coolant flow (cycle): {jd['mdot_total']:.2f} kg/s total → "
              f"{bold('%.2f kg/s' % jd['mdot_cool'])} coolant "
              f"{dim('(fuel, MR=%g)' % args.mixture_ratio)}")
        if jd["channel_count"] is None:
            print(red("    no channel geometry fits the throat; reduce the count."))
            return 2
        b = jd["band"]
        if jd["feasible"]:
            if jd["fatigue_status"] in {
                "design_qualified_gate", "sourced_screening_gate"
            }:
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
            channel_height=args.channel_height,
            channel_roughness=args.channel_roughness,
            gate_coolant_chemistry=args.gate_coolant_chemistry,
            curvature_correction=args.curvature_correction,
            mixture_ratio=args.mixture_ratio,
            cooling_fraction=args.cooling_fraction, coolant=args.coolant,
            thermal_margin=args.margin_target, structural_fos=args.structural_fos,
            helix_turns=args.helix_turns,
            t_hot_min=args.t_hot_min, t_hot_max=args.t_hot_max,
            channel_height_min=args.channel_height_min,
            channel_height_max=args.channel_height_max,
            n_channel_height=args.channel_height_steps,
            dp_budget_bar=args.dp_budget,
            buckling_fos=args.buckling_fos,
            buckling_tangent_modulus_fraction=
                args.buckling_tangent_modulus_fraction,
            gate_sp125_429=args.gate_sp125_tube_buckling,
            coolant_outlet_pressure=args.coolant_outlet_pressure,
            injector_pressure_drop=args.injector_pressure_drop,
            cooling_options=args._cooling_options)
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
            "sp125_429_temperature_status":
                prof["sp125_429_temperature_status"],
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
        from raosim.thermal_design import size_cooling_channels
        prop = args._prop
        sized = size_cooling_channels(
            contour, prop, args.pc,
            margin_target=args.margin_target, dp_budget_bar=args.dp_budget,
            wall_temp_limit=args.wall_temp_limit,
            mixture_ratio=args.mixture_ratio,
            cooling_fraction=args.cooling_fraction,
            channel_height=args.channel_height, wall_thickness=args.wall_thickness,
            channel_roughness=args.channel_roughness,
            gate_coolant_chemistry=args.gate_coolant_chemistry,
            curvature_correction=args.curvature_correction,
            wall_k=args.wall_k, coolant=args.coolant,
            helix_turns=args.helix_turns, objective=args.size_objective,
            cooling_options=args._cooling_options)
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
            channel_pressure_hoop_radius,
            coaxial_shell_wall_stress_profile,
            regenerative_cooling_analysis,
        )
        prop = args._prop
        spec = CoolingSpec(
            method="regenerative", coolant=args.coolant,
            channel_count=args.channels,
            channel_width=args.channel_width,
            channel_height=args.channel_height,
            channel_roughness=args.channel_roughness,
            coolant_mass_flow=args.coolant_mdot,
            coolant_outlet_pressure=args.coolant_outlet_pressure,
            injector_pressure_drop=args.injector_pressure_drop,
            max_wall_temperature=args.wall_temp_limit,
            **args._cooling_options,
        )
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
            curvature_correction=args.curvature_correction,
            coolant_outlet_pressure=args.coolant_outlet_pressure,
            injector_pressure_drop=args.injector_pressure_drop)
        pf = cooling_result.get("passage_length_factor", 1.0)
        summary["cooling"] = {
            "peak_wall_T_K": cooling_result["peak_gas_side_wall_temperature"],
            "cooling_margin": cooling_result["cooling_margin"],
            "coolant_outlet_T_K": cooling_result["coolant_outlet_temperature"],
            "coolant_inlet_T_K": args.coolant_inlet_temperature,
            "coolant_inlet_temperature_source":
                args._coolant_inlet_temperature_source,
            "coolant_chemistry_margin":
                cooling_result["coolant_chemistry_margin"],
            "coolant_chemistry_status":
                cooling_result["coolant_chemistry_status"],
            "pressure_drop_bar": cooling_result["coolant_pressure_drop"] / 1e5,
            "channel_roughness_m": cooling_result["channel_roughness"],
            "pressure_drop_correlation":
                cooling_result["pressure_drop_correlation"],
            "peak_channel_velocity_m_s":
                cooling_result["peak_channel_velocity"],
            "gas_radiation_status":
                cooling_result["gas_radiation_status"],
            "coolant_property_backend":
                cooling_result["coolant_property_backend"],
            "coolant_property_status":
                cooling_result["coolant_property_status"],
            "hydraulic_network_status":
                cooling_result["hydraulic_network_status"],
            "channel_friction_pressure_drop_bar":
                cooling_result["channel_friction_pressure_drop"] / 1e5,
            "boiling_chf_status":
                cooling_result["boiling_chf_status"],
            "curvature_correlation_status":
                cooling_result["curvature_correlation_status"],
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
        # The defining regen image: Bartz heat flux, the SP-125 series-circuit
        # wall solution (T_wg/T_wc) and the Naraghi coolant march, with the
        # material gas-side limit and the SP-8087 coking limit drawn.
        try:
            from raosim.plotting import plot_cooling_profile
            fig = plot_cooling_profile(
                cooling_result, contour=contour,
                max_wall_temperature=getattr(material, "max_temperature", None),
                save_path=args.out / "cooling_profile.png", show=show)
            fig.clf()
            artifacts.append("cooling_profile.png")
            print(green("    wrote cooling_profile.png"))
            from raosim.plotting import plot_coolant_channel_march
            fig = plot_coolant_channel_march(
                cooling_result, save_path=args.out / "channel_march.png",
                show=show)
            fig.clf()
            artifacts.append("channel_march.png")
            print(green("    wrote channel_march.png"))
        except Exception as exc:  # a plotting hiccup must never break the run
            print(yellow(f"    cooling plots skipped: {exc}"))
        # Channel-land cross-section temperature map at the gas-side hot
        # spot — the circumferential land peak the 1-D circuit averages away.
        try:
            from raosim.physics import wall_cross_section_field
            from raosim.plotting import plot_channel_cross_section
            xs = wall_cross_section_field(
                cooling_result, hf, contour, spec, material,
                (analysis_wall_profile.t_hot
                 if analysis_wall_profile is not None else args.wall_thickness),
                prop, args.pc, station="peak")
            fig = plot_channel_cross_section(
                xs, save_path=args.out / "cross_section.png", show=show)
            fig.clf()
            artifacts.append("cross_section.png")
            print(green("    wrote cross_section.png")
                  + dim("  (land–channel ΔT %.0f K)"
                        % xs["circumferential_spread"]))
        except Exception as exc:
            print(yellow(f"    cross_section.png skipped: {exc}"))
        if cooling_result["coolant_property_backend"] == "CoolProp_HEOS":
            print(dim(
                "    coolant properties: CoolProp HEOS, station-wise T/p "
                "iteration"
            ))
        if cooling_result["hydraulic_network_status"] == "full_channel_graph_converged":
            net = cooling_result["hydraulic_network"]
            print(dim(
                f"    hydraulic network: channel friction "
                f"{cooling_result['channel_friction_pressure_drop']/1e5:.1f} bar, "
                f"source→sink {cooling_result['coolant_pressure_drop']/1e5:.1f} bar, "
                f"flow spread {100*net['maldistribution_fraction']:.2f}%"
            ))
        if cooling_result["gas_radiation_status"] != "not_included":
            print(dim(
                f"    radiation: {cooling_result['gas_radiation_status']}, "
                f"peak {max(cooling_result['radiative_heat_flux'])/1e6:.3f} MW/m²"
            ))
        if cooling_result["boiling_chf_status"] != "disabled":
            print(dim(
                f"    boiling/CHF: {cooling_result['boiling_chf_status']}"
            ))
        if not cooling_result["coolant_chemistry_feasible"]:
            print(yellow(
                "    coolant chemistry: RP-1/kerosene coolant-side wall "
                "exceeds the conservative 700 K coking screen"
            ))
        for warning in cooling_result["warnings"]:
            print(yellow("    warning: " + warning))

        # ---- structural screen: SP-125 eq. 4-31 combined wall stress -----
        # Needs the elastic/thermal properties, which only a catalog
        # --material supplies.  The thin liner carries the station-wise
        # COOLANT-GAS differential from the hydraulic and gas-pressure
        # marches, not a fixed fraction of chamber pressure.
        if material.elastic_modulus:
            t_wall_struct = (
                analysis_wall_profile.t_hot
                if analysis_wall_profile is not None
                else args.wall_thickness
            )
            stress = coaxial_shell_wall_stress_profile(
                pressure_differential=cooling_result["liner_pressure_differential"],
                # SP-125 eq. 4-27 hoop radius = channel tube radius, NOT the
                # nozzle shell radius contour["y"] (which overstated the liner
                # pressure stress by ~1000x at the exit).
                inner_radius=channel_pressure_hoop_radius(
                    args.channel_width, t_wall_struct),
                wall_thickness=t_wall_struct,
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
            # Structural + life dashboard: eq. 4-31 stress vs yield and the
            # sourced low-cycle fatigue life N_f(x) (NARloy-Z CR-134627 /
            # GRCop-84 Lerch-Ellis) along the wall.
            try:
                from raosim.plotting import plot_structural_life_dashboard
                fig = plot_structural_life_dashboard(
                    cooling_result, stress, material=material,
                    required_cycles=getattr(args, "required_cycles", None),
                    save_path=args.out / "structural_life.png", show=show)
                fig.clf()
                artifacts.append("structural_life.png")
                print(green("    wrote structural_life.png"))
            except Exception as exc:  # plotting must never break the run
                print(yellow(f"    structural_life.png skipped: {exc}"))

    # ---- pintle injector  (sized from the operating-point ṁ split) ------
    # Placed AFTER the cooling analysis so the fuel feed state (T, P -> rho/mu/
    # sigma) comes from the regen jacket OUTLET when the fuel is the coolant:
    # cooling -> injector feed -> hydraulics.  Failing gates block the chamber
    # export below unless --allow-infeasible-injector is set.
    if args.injector == "pintle":
        from raosim.injector import (
            InjectorSpec, PropellantFeedSpec, PintleGeometrySpec,
            InjectorManufacturingSpec, InjectorUnsupportedState,
            InjectorSpecError, evaluate_pintle_injector,
        )
        from raosim.design import CoolingSpec
        print("\n" + cyan("▸ " + bold("Injector")) +
              dim("  (pintle, liquid/liquid, sized from the ṁ split)"))
        if not args._ox_name or not args._fuel_name:
            print(red("    --injector pintle needs real propellant identities; "
                      "pass --oxidizer/--fuel or --propellant 'OX/FUEL'."))
            return 2
        inj_spec = InjectorSpec(
            type="pintle", sizing=args.injector_sizing,
            fuel_dp_fraction=args.fuel_injector_dp_fraction,
            oxidizer_dp_fraction=args.oxidizer_injector_dp_fraction,
            fuel_cd=args.fuel_discharge_coefficient,
            oxidizer_cd=args.oxidizer_discharge_coefficient,
            faceplate_material=args.material, pintle_material=args.material,
            target_momentum_ratio=args.pintle_target_momentum_ratio,
            allow_infeasible=args.allow_infeasible_injector,
            fuel=PropellantFeedSpec(
                role="fuel", name=args._fuel_name,
                inlet_temperature=args.fuel_inlet_temperature,
                inlet_pressure=args.fuel_inlet_pressure,
                phase=args.fuel_phase),
            oxidizer=PropellantFeedSpec(
                role="oxidizer", name=args._ox_name,
                inlet_temperature=args.oxidizer_inlet_temperature,
                inlet_pressure=args.oxidizer_inlet_pressure,
                phase=args.oxidizer_phase),
            geometry=PintleGeometrySpec(
                pintle_diameter=args.pintle_diameter,
                slot_count=args.pintle_slot_count,
                slot_aspect_ratio=args.pintle_slot_aspect_ratio,
                deflector_angle=args.pintle_deflector_angle,
                impingement_distance=args.pintle_impingement_distance,
                radial_stream=args.pintle_radial_stream,
                annulus_gap=args.pintle_annulus_gap,
                slot_width=args.pintle_slot_width,
                slot_height=args.pintle_slot_height,
                slot_depth=args.pintle_slot_depth,
                tip_radius=args.pintle_tip_radius,
                body_length=args.pintle_body_length,
                face_thickness=args.injector_face_thickness,
                face_od=args.injector_face_od),
            manufacturing=InjectorManufacturingSpec(
                min_feature=args.injector_min_feature),
        )
        try:
            coupling_cooling = CoolingSpec(
                method="regenerative" if args.regen else "none",
                coolant=args.coolant,
                coolant_mass_flow=args.coolant_mdot,
            )
            inj = evaluate_pintle_injector(
                inj_spec,
                mdot_fuel=args._mdot_f,
                mdot_oxidizer=args._mdot_o,
                Pc=args.pc, mixture_ratio=args.mixture_ratio,
                chamber_radius=chamber["Rc"], chamber_length=chamber["Lc"],
                gamma=prop.gamma, Tc=prop.Tc, R_gas=prop.R_gas,
                fuel_name=args._fuel_name, oxidizer_name=args._ox_name,
                cooling=coupling_cooling, cooling_result=cooling_result,
            )
        except (InjectorUnsupportedState, InjectorSpecError) as exc:
            print(red(f"    injector rejected: {exc}"))
            summary["injector"] = {"type": "pintle", "feasible": False,
                                   "rejected": str(exc)}
            (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
            return 2
        _print_injector_panel(inj)
        coupling_gate = next(
            gate for gate in inj.gates
            if gate.name == "regen_fuel_flow_closure"
        )
        print(dim(f"    coupling: {coupling_gate.detail}"))
        inj_dict = inj.to_dict()
        summary["injector"] = inj_dict
        # Standalone pintle build artifact (sizing + streams + atomization +
        # gates), written alongside summary.json regardless of feasibility.
        (args.out / "pintle.json").write_text(json.dumps(inj_dict, indent=2))
        artifacts.append("pintle.json")
        print(green("    wrote pintle.json"))
        if not inj.feasible and not args.allow_infeasible_injector:
            print(red("    injector gates FAILED — blocking chamber export; "
                      "re-run with --allow-infeasible-injector to override."))
            (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
            return 2
        print(green("    injector sized ✓") if inj.feasible else
              yellow("    injector sized with FAILING gates "
                     "(--allow-infeasible-injector)"))

        # ---- optional movable-sleeve throttle map (computed before the
        #      figures so it can be plotted alongside them) --------------
        tm = None
        if args.throttle_map:
            from raosim.injector import throttle_map
            try:
                levels = tuple(sorted(float(x) for x in
                                      args.throttle_map.split(",") if x.strip()))
                tm = throttle_map(
                    inj_spec, mdot_fuel_full=args._mdot_f,
                    mdot_oxidizer_full=args._mdot_o, Pc_full=args.pc,
                    mixture_ratio=args.mixture_ratio,
                    chamber_radius=chamber["Rc"], chamber_length=chamber["Lc"],
                    gamma=prop.gamma, Tc=prop.Tc, R_gas=prop.R_gas,
                    levels=levels, pc_exponent=args.throttle_pc_exponent)
                summary["injector_throttle_map"] = tm.to_dict()
                print("\n    " + bold("Throttle map")
                      + dim(f"  (Pc∝f^{args.throttle_pc_exponent:g}; "
                            "O/F+TMR held by the sleeve)"))
                print(f"      {'f':>5} {'Pc[bar]':>8} {'stroke':>7} "
                      f"{'v_a':>6} {'TMR':>6} {'SMD[µm]':>8} {'η_c*':>6} "
                      f"{'feas':>5}")
                for p in tm.points:
                    badge = green("yes") if p.feasible else red("no")
                    print(f"      {p.throttle:>5.2f} {p.Pc/1e5:>8.1f} "
                          f"{p.sleeve_stroke_fraction:>7.3f} {p.v_annulus:>6.0f} "
                          f"{p.total_momentum_ratio:>6.3f} "
                          f"{p.smd_limiting*1e6:>8.0f} "
                          f"{p.predicted_cstar_efficiency:>6.2f} {badge:>5}")
            except Exception as exc:
                print(yellow(f"    throttle map skipped: {exc}"))

        # ---- pintle deliverable package (mandatory schematic + table) ---
        try:
            from raosim.injector_export import export_pintle_package

            cad_mode = args.injector_cad
            cad_format = args.injector_cad_format
            if args.injector_cad == "step":
                cad_mode = "parts"
                cad_format = "step"
            elif args.injector_cad == "auto":
                cad_mode = "parts"

            pkg = export_pintle_package(
                inj, args.out / "pintle", spec=inj_spec,
                cad=cad_mode, cad_format=cad_format,
                movable_sleeve=args.pintle_sleeve)
            summary["injector_package"] = {
                "dir": pkg["dir"],
                "files": pkg["files"],
                "notes": pkg["notes"],
            }
            for p in pkg["files"].values():
                artifacts.append(os.path.relpath(str(p), str(args.out)))
            print(green(f"    wrote pintle/ package "
                        f"({len(pkg['files'])} files)"))
            for note in pkg["notes"]:
                msg = f"    pintle package: {note}"
                print(red(msg) if args.injector_cad == "step" else yellow(msg))
        except Exception as exc:
            print(yellow(f"    pintle package skipped: {exc}"))

        # ---- injector diagnostic figures (full set) ------------------
        try:
            from raosim.injector_plots import export_all_injector_figures
            figs = export_all_injector_figures(
                inj, args.out, show=show, throttle=tm)
            artifacts += figs
            print(green(f"    wrote {len(figs)} injector diagnostic PNGs "
                        + dim("(" + ", ".join(
                            f.replace("injector_", "").replace(".png", "")
                            for f in figs) + ")")))
        except Exception as exc:
            print(yellow(f"    injector figures skipped: {exc}"))

    # ---- injector/chamber mechanical interface screen -----------------
    try:
        from raosim.interface import screen_injector_chamber_interface

        interface_ledger = screen_injector_chamber_interface(
            chamber_pressure=args.pc,
            chamber_radius=chamber["Rc"],
            wall_thickness=args.wall_thickness,
            face_outer_diameter=args.injector_face_od,
            face_thickness=args.injector_face_thickness,
            flange_outer_diameter=args.flange_od,
            flange_length=args.flange_length,
            bolt_count=args.bolt_count,
            bolt_circle_diameter=args.bolt_circle,
            bolt_hole_diameter=args.bolt_hole,
            bolt_diameter=args.bolt_diameter,
            bolt_allowable_stress=args.bolt_allowable_stress,
            material_yield_strength=(mat.yield_strength if mat else None),
            material_elastic_modulus=(mat.elastic_modulus if mat else None),
            material_poisson_ratio=(mat.poisson_ratio if mat else None),
            joint_separation_factor=args.joint_separation_factor,
        )
        summary["injector_interface"] = interface_ledger.to_dict()
        if args.injector == "pintle" or any(
            v is not None for v in (
                args.injector_face_od, args.injector_face_thickness,
                args.flange_od, args.bolt_count, args.bolt_circle,
                args.bolt_hole, args.bolt_diameter,
            )
        ):
            failed = [g for g in interface_ledger.gates if g.status == "fail"]
            warnish = [g for g in interface_ledger.gates if g.status == "info"]
            badge = red("FAIL") if failed else (
                yellow("screen") if warnish else green("pass")
            )
            print("\n" + cyan("▸ " + bold("Injector interface"))
                  + dim("  (pressure load + faceplate + bolt pattern)"))
            print(
                f"    pressure separating load: "
                f"{interface_ledger.separating_force/1000:.1f} kN   "
                f"status: {badge}"
            )
            for gate in failed[:3]:
                print(red(f"    {gate.name}: {gate.detail}"))
            for gate in warnish[:2]:
                print(yellow(f"    {gate.name}: {gate.detail}"))
    except Exception as exc:
        print(yellow(f"    injector interface screen skipped: {exc}"))

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
    wall_thickness_mode = (
        "station_wise_thermostructural_sizing"
        if sized_wall_profile is not None
        else "uniform_reference_input_not_sized"
    )
    if sized_wall_profile is None:
        print(yellow(
            "\n    wall thickness is a uniform reference input, not a "
            "thermostructural design. Use --regen --material <alloy> "
            "--size-wall for station-wise liner and jacket sizing."
        ))
    wall_path = export_stl(
        x, y, args.out / "wall.stl", n_angular=96,
        wall_thickness=wall_thickness_geometry,
        flange_od=args.flange_od,
        flange_length=args.flange_length,
    )
    wall_mesh = inspect_stl(wall_path)
    print(green(
        f"    wall STL: watertight solid, "
        f"{wall_mesh['volume_m3'] * 1e3:.6g} L, "
        f"{wall_mesh['boundary_edge_count']} boundary edges"
    ))
    artifacts.append(wall_path.name)
    summary["wall_geometry"] = {
        "uniform_seed_thickness_m": float(args.wall_thickness),
        "uniform_seed_source": args._wall_thickness_source,
        "selected_sizing_mode": args._wall_sizing_mode,
        "thickness_mode": wall_thickness_mode,
        "t_hot_range_m": [
            float(np.min(np.asarray(wall_thickness_geometry))),
            float(np.max(np.asarray(wall_thickness_geometry))),
        ],
        "offset": "surface_normal",
        "stl": "closed_solid_triangle_mesh",
        "stl_watertight": wall_mesh["watertight"],
        "stl_boundary_edge_count": wall_mesh["boundary_edge_count"],
        "stl_nonmanifold_edge_count": wall_mesh["nonmanifold_edge_count"],
        "stl_volume_m3": wall_mesh["volume_m3"],
        "wall_scope": "liner_base_only_no_channel_ribs_or_jacket",
        "flange_od_m": args.flange_od,
        "flange_length_m": args.flange_length,
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
                flange_od=args.flange_od,
                flange_length=args.flange_length,
                require_brep=args.require_brep,
                metadata={
                    "uniform_seed_thickness_m": args.wall_thickness,
                    "thickness_mode": wall_thickness_mode,
                    "t_hot_range_m":
                        summary["wall_geometry"]["t_hot_range_m"],
                    "flange_od_m": args.flange_od,
                    "flange_length_m": args.flange_length,
                    "material": mat.name if mat else None,
                    "hardware_qualified": False,
                },
                throat_location=contour["throat_location"],
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
                    "network_overlaps": rb["network_overlaps"],
                    "fuse_healing": rb["fuse_healing"],
                    "cut_healing": rb["cut_healing"],
                    "final_healing": rb["final_healing"],
                    "shape_fix": rb["shape_fix"],
                    "step_precision_mode": rb["step_precision_mode"],
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
                "uniform_seed_thickness_m": args.wall_thickness,
                "thickness_mode": wall_thickness_mode,
                "t_hot_range_m": summary["wall_geometry"]["t_hot_range_m"],
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
