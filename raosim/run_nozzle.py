"""
run_nozzle.py — the main runner: solve the contour with the
differentiable / MOC backend, then (optionally) add the regen cooling
coils, run the cooling analysis, and export everything.

Pipeline
--------
1. Construct the trusted deterministic Rao/TOP chart Bézier contour by
   default.  ``--contour-method rao-bvp`` opts into the experimental Rao
   variational / MOC boundary-value problem (``solve_rao_bvp``):
   ``--backend jax`` is the differentiable Optimistix-LM path and
   ``--backend numpy`` is the SciPy finite-difference oracle.
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

    # experimental BVP contour, host (full quality):
    PYTHONPATH=. python scripts/run_nozzle.py --contour-method rao-bvp --backend jax \
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

Note: the full ``--contour-method rao-bvp --backend jax`` solve
(max_nfev ~4000 + weight ladder)
runs for minutes — it is a host job. ``--max-nfev 0`` evaluates the seed
only and is useful for diagnostics; hard geometry-gate failures block export.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
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


def _parse_opening_cd_map(value: str) -> tuple[tuple[float, float], ...]:
    """Parse ``opening_fraction:Cd`` pairs for movable-pintle calibration."""

    points: list[tuple[float, float]] = []
    try:
        for item in str(value).split(","):
            fraction_text, cd_text = item.strip().split(":", 1)
            points.append((float(fraction_text), float(cd_text)))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated opening_fraction:Cd pairs, e.g. "
            "'0:0.62,0.5:0.70,1:0.76'"
        ) from exc
    if len(points) < 2:
        raise argparse.ArgumentTypeError(
            "movable-pintle Cd map requires at least two points"
        )
    return tuple(points)


_LOGO = r"""
   ██╗     ██████╗ ███████╗      LREKit
   ██║     ██╔══██╗██╔════╝      liquid rocket engine toolkit
   ██║     ██████╔╝█████╗        ───────────────────────────
   ██║     ██╔══██╗██╔══╝        nozzle · chamber · injector
   ███████╗██║  ██║███████╗      regen cooling · CAD export
   ╚══════╝╚═╝  ╚═╝╚══════╝
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
        ("Default package", [
            ("bare run / --complete-package",
             "screen nozzle + chamber + pintle injector + electric pump"),
            ("--injector {pintle,none}", "selected injector package"),
            ("--electric-pump / --no-electric-pump",
             "selected fuel-pump package"),
            ("--propellant 'LOX/RP-1'", "default complete-package propellant"),
        ]),
        ("Propellant / operating point", [
            ("--propellant / --oxidizer / --fuel",
             "combustion pair and injector feed identities"),
            ("--thermo-mode {constant-gamma,cea}",
             "built-in screening table or RocketCEA chamber snapshot"),
            ("--nozzle-expansion-model {constant-gamma,frozen-variable-cp}",
             "calorically perfect or fixed-composition variable-cp Q1D"),
            ("--frozen-gas-table", "strict provenance-bound cp(T) JSON table"),
            ("--pc / --pa-over-p0", "chamber pressure [Pa] / ambient ratio"),
            ("--target-thrust / --rt", "size throat from thrust, or set Rt"),
            ("--mixture-ratio", "O/F flow split; default from propellant table"),
            ("--eta-cstar / --eta-cf", "combustion and nozzle efficiency overrides"),
        ]),
        ("Nozzle", [
            ("--contour-method {bezier,rao-bvp}",
             "trusted Rao/TOP chart contour (default) or experimental BVP"),
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
            ("--shoulder-sizing {scalar,auto}", "manual shoulder or geometric closure"),
            ("--minimum-cylindrical-length", "minimum useful cylinder [m]"),
            ("--ru-factor / --cd-target", "upstream throat radius or target inviscid Cd"),
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
            ("--coolant-outlet-pressure",
             "absolute jacket outlet pressure override [Pa]"),
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
        ("Pintle injector", [
            ("--injector {pintle,none}", "generate/evaluate the selected injector"),
            ("--injector-architecture {fixed_discrete,son_continuous_movable}",
             "fixed openings or Son continuous moving centre rod"),
            ("--injector-sizing {auto,fixed,movable}",
             "auto-size, evaluate fixed geometry, or hold annulus and solve travel"),
            ("--fuel/oxidizer-injector-dp-fraction",
             "metering pressure drop as ΔP/Pc"),
            ("--fuel/oxidizer-discharge-coefficient",
             "per-stream metering Cd"),
            ("--pintle-radial-stream {fuel,oxidizer}",
             "which stream feeds the radial slots"),
            ("--pintle-diameter / --pintle-slot-count",
             "pintle anchor diameter and radial openings"),
            ("--pintle-slot-aspect-ratio / --pintle-deflector-angle",
             "slot h/w and spray deflection"),
            ("--pintle-target-momentum-ratio",
             "optional TMR gate for radial/axial momentum balance"),
            ("--pintle-annulus-gap / --pintle-slot-width/height/depth",
             "fixed-geometry passage overrides"),
            ("--injector-face-od / --injector-face-thickness",
             "faceplate packaging and interface screen"),
            ("--bolt-count / --bolt-circle / --bolt-hole / --bolt-diameter",
             "injector-to-chamber bolted joint"),
            ("--fuel/oxidizer-inlet-count/diameter/angle",
             "feed-port layout and inlet velocity screens"),
            ("--fuel/oxidizer-manifold-width/depth",
             "annular manifold layout screens"),
            ("--injector-cad / --injector-cad-format / --pintle-sleeve",
             "pintle deliverable package"),
            ("--movable-post-* / --movable-center-gap-* / --movable-cd-map",
             "Son geometry and configuration-controlled hydraulic calibration"),
            ("--movable-position-* / --movable-actuator-* / --movable-stem-*",
             "metrology and static actuation evidence"),
            ("--movable-sheet-thickness-*",
             "separate VOF/measured sheet handoff evidence"),
        ]),
        ("Electric pump", [
            ("--electric-pump", "size electric pump drive, battery, and pump geometry"),
            ("--no-electric-pump", "disable the default complete-package pump"),
            ("--feed-architecture {pump_fed,pressure_fed}",
             "feed ledger label"),
            ("--fuel/oxidizer-tank-pressure",
             "pump inlet pressure for head, power, and NPSH"),
            ("--fuel/oxidizer-supply-pressure",
             "available pump/tank outlet pressure gate"),
            ("--fuel/oxidizer-flow-capacity",
             "available pump/feed mass-flow capacity gate"),
            ("--fuel/oxidizer-line-loss[-fraction]",
             "line, valve, and filter losses charged to pump"),
            ("--fuel/oxidizer-manifold-loss[-fraction]",
             "declared manifold allowance charged once to pump"),
            ("--pump-rpm / --burn-time", "pump speed [rpm] / burn duration [s]"),
            ("--motor-voltage / --inverter-power-density", "drive bus / inverter sizing"),
            ("--battery-energy-density / --battery-power-density",
             "pack-level energy [J/kg] / pulse power [W/kg]"),
            ("--pump-head-coefficient / --pump-flow-coefficient",
             "centrifugal meanline design coefficients"),
            ("--pump-tip-speed-limit / --pump-max-head-per-stage",
             "geometry and staging screens"),
            ("--pump-cad / --pump-cad-format",
             "pump part CAD package: impeller, inducer, diffuser, drive"),
            ("--allow-open-pump-mesh",
             "waive the pump part STL watertightness gate"),
            ("--pump-visualize", "save a simplified pump particle GIF"),
        ]),
        ("Visualisation", [
            ("--flowfield", "render resolved MOC Mach/p/theta/T fields"),
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
            if len(flag) > 41:
                print("    " + green(flag))
                print("      " + dim(desc))
            else:
                print("    " + green(f"{flag:<42}") + dim(desc))
    print()

import numpy as np

plt = None


def _want_windows(args=None, argv: list[str] | None = None) -> bool:
    """True when plots should use a live backend instead of file-only Agg."""
    argv = sys.argv[1:] if argv is None else argv
    interactive_run = (
        len(argv) == 0 or "-i" in argv or "--interactive" in argv
    )
    requested_show = bool(getattr(args, "show", False)) if args is not None else (
        "--show" in argv
    )
    return bool((requested_show or interactive_run) and sys.stdout.isatty())


def _ensure_pyplot(show: bool):
    """Import Matplotlib lazily so parse-only paths such as --help stay light."""
    global plt
    if plt is None:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        plt = _plt
    return plt

import raosim.rao_variational as rv
from raosim.export import (
    export_step,
    export_stl,
    inspect_stl,
    package_ipt_request,
    step_representation,
)
from raosim.chamber_geometry import (
    auto_shoulder_factor,
    chamber_contour,
    failed_thrust_chamber_geometry_checks,
    full_engine_contour,
)
from raosim.materials import get_material, material_names, material_table
from raosim.physics import default_coolant_inlet_temperature
from raosim.rao_variational import RaoSolverConfig
from raosim.regen_geometry import generate_regen_nozzle
from raosim.throat_geometry import (
    REPOSITORY_UPSTREAM_RADIUS_RATIO_EXTENSION_BOUNDS,
    SP8120_UPSTREAM_RADIUS_RATIO_BOUNDS,
    ThroatGeometrySpec,
    throat_discharge_coefficient_hall,
    upstream_radius_ratio_for_discharge_coefficient,
)
from raosim.coolants import canonical_coolant_name


_DEFAULT_INJECTOR_DP_FRACTION = 0.2
_DEFAULT_COMPLETE_PROPELLANT = "LOX/RP-1"
_DEFAULT_COMPLETE_FUEL_TANK_PRESSURE = 5.0e5
_DEFAULT_COMPLETE_OXIDIZER_TANK_PRESSURE = 6.0e5


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


def _prompt_optional_float(label, default):
    shown = "auto" if default is None else default
    raw = input(f"  {label} [{shown}]: ").strip()
    if raw == "":
        return default
    if raw.lower() in {"auto", "none", "blank", "-"}:
        return None
    try:
        return float(raw)
    except ValueError:
        print(f"    (couldn't parse {raw!r}; keeping {shown})")
        return default


def _prompt_optional_str(label, default):
    shown = "blank" if default is None else default
    raw = input(f"  {label} [{shown}]: ").strip()
    if raw == "":
        return default
    if raw.lower() in {"none", "blank", "-"}:
        return None
    return raw


def _prompt_choice(label, default, choices):
    raw = input(f"  {label} [{default}]: ").strip()
    if raw == "":
        return default
    value = raw.lower()
    if value in choices:
        return value
    print(f"    (choose one of {', '.join(choices)}; keeping {default})")
    return default


def _prompt_bool(label, default):
    raw = input(f"  {label} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    if raw == "":
        return default
    return raw.startswith("y")


def _section(title: str) -> None:
    print("\n" + cyan("▸ " + bold(title)))


def _write_moc_crossing_samples(path: Path, samples: list[dict]) -> None:
    """Write a flat CSV table for the sampled MOC characteristic crossings."""
    fields = [
        "crossing_index",
        "intersection_x_m",
        "intersection_r_m",
        "segment_1_family",
        "segment_1_row",
        "segment_1_role",
        "segment_1_parent_index",
        "segment_1_child_index",
        "segment_1_parent_x_m",
        "segment_1_parent_r_m",
        "segment_1_child_x_m",
        "segment_1_child_r_m",
        "segment_2_family",
        "segment_2_row",
        "segment_2_role",
        "segment_2_parent_index",
        "segment_2_child_index",
        "segment_2_parent_x_m",
        "segment_2_parent_r_m",
        "segment_2_child_x_m",
        "segment_2_child_r_m",
    ]

    def row_for(sample: dict) -> dict:
        s1 = sample.get("segment_1", {}) or {}
        s2 = sample.get("segment_2", {}) or {}
        p1 = s1.get("parent", {}) or {}
        c1 = s1.get("child", {}) or {}
        p2 = s2.get("parent", {}) or {}
        c2 = s2.get("child", {}) or {}
        ix = sample.get("intersection", {}) or {}
        return {
            "crossing_index": sample.get("crossing_index"),
            "intersection_x_m": ix.get("x"),
            "intersection_r_m": ix.get("r"),
            "segment_1_family": s1.get("family"),
            "segment_1_row": s1.get("row"),
            "segment_1_role": s1.get("role"),
            "segment_1_parent_index": s1.get("parent_index"),
            "segment_1_child_index": s1.get("child_index"),
            "segment_1_parent_x_m": p1.get("x"),
            "segment_1_parent_r_m": p1.get("r"),
            "segment_1_child_x_m": c1.get("x"),
            "segment_1_child_r_m": c1.get("r"),
            "segment_2_family": s2.get("family"),
            "segment_2_row": s2.get("row"),
            "segment_2_role": s2.get("role"),
            "segment_2_parent_index": s2.get("parent_index"),
            "segment_2_child_index": s2.get("child_index"),
            "segment_2_parent_x_m": p2.get("x"),
            "segment_2_parent_r_m": p2.get("r"),
            "segment_2_child_x_m": c2.get("x"),
            "segment_2_child_r_m": c2.get("r"),
        }

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow(row_for(sample))


def _default_coolant_inlet_temperature(coolant: str) -> float:
    """Backward-compatible alias for the central physics-layer resolver."""
    return default_coolant_inlet_temperature(coolant)


def _argument_present(argv: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(option + "=") for arg in argv)


def _any_argument_present(argv: list[str], options: tuple[str, ...]) -> bool:
    return any(_argument_present(argv, option) for option in options)


def _expand_arg_files(
    argv: list[str],
    *,
    base_dir: Path | None = None,
    seen: frozenset[Path] = frozenset(),
) -> list[str]:
    """Expand ``@path`` arguments into shell-style CLI tokens.

    Arg files are intentionally simple: blank lines and ``#`` comments are
    ignored, quoting follows POSIX shell rules, and nested ``@`` files are
    resolved relative to the file that includes them.
    """
    expanded: list[str] = []
    for arg in argv:
        if not (arg.startswith("@") and len(arg) > 1):
            expanded.append(arg)
            continue

        path = Path(arg[1:])
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        path = path.resolve()
        if path in seen:
            raise ValueError(f"recursive argument file include: @{path}")
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"could not read argument file @{path}: {exc}") from exc

        file_args: list[str] = []
        for lineno, line in enumerate(text.splitlines(), start=1):
            try:
                file_args.extend(shlex.split(line, comments=True, posix=True))
            except ValueError as exc:
                raise ValueError(
                    f"could not parse argument file @{path}:{lineno}: {exc}"
                ) from exc
        expanded.extend(
            _expand_arg_files(
                file_args, base_dir=path.parent, seen=seen | frozenset({path})
            )
        )
    return expanded


def _apply_complete_package_defaults(args, argv: list[str], *, reason: str) -> None:
    """Turn the accessible default package into concrete backend inputs.

    This keeps batch compatibility: explicit user flags always win, while a
    bare or ``--complete-package`` run has enough data to produce the nozzle,
    selected pintle injector, and selected electric pump geometry.
    """
    notes = getattr(args, "_complete_package_default_notes", [])
    prop_explicit = _any_argument_present(
        argv,
        ("--propellant", "--oxidizer", "--fuel"),
    )
    if not prop_explicit and args.propellant is None:
        args.propellant = _DEFAULT_COMPLETE_PROPELLANT
        notes.append(f"propellant={args.propellant}")

    injector_explicit = _argument_present(argv, "--injector")
    if not injector_explicit and args.injector == "none":
        args.injector = "pintle"
        notes.append("injector=pintle")

    electric_explicit = _any_argument_present(
        argv, ("--electric-pump", "--no-electric-pump")
    )
    if not electric_explicit:
        args.electric_pump = args.injector == "pintle"
        if args.electric_pump:
            notes.append("electric_pump=on")

    if args.electric_pump:
        if (
            not _argument_present(argv, "--fuel-tank-pressure")
            and args.fuel_tank_pressure is None
        ):
            args.fuel_tank_pressure = _DEFAULT_COMPLETE_FUEL_TANK_PRESSURE
            notes.append(
                f"fuel_tank_pressure={args.fuel_tank_pressure:g} Pa"
            )
        if (
            not _argument_present(argv, "--oxidizer-tank-pressure")
            and args.oxidizer_tank_pressure is None
        ):
            args.oxidizer_tank_pressure = (
                _DEFAULT_COMPLETE_OXIDIZER_TANK_PRESSURE
            )
            notes.append(
                f"oxidizer_tank_pressure={args.oxidizer_tank_pressure:g} Pa"
            )

    args._complete_package_defaults = bool(notes)
    args._complete_package_reason = reason
    args._complete_package_default_notes = notes


def _reject_legacy_injector_pressure_drop(parser, argv: list[str]) -> None:
    if _argument_present(argv, "--injector-pressure-drop"):
        parser.error(
            "--injector-pressure-drop is deprecated and no longer controls "
            "injector sizing or regen pressure boundaries. Use "
            "--fuel-injector-dp-fraction and --oxidizer-injector-dp-fraction "
            "for the split injector model, or --coolant-outlet-pressure for "
            "an explicit absolute jacket outlet pressure."
        )


def _coolant_is_cycle_fuel(coolant: str | None, fuel_name: str | None) -> bool:
    return bool(
        coolant
        and fuel_name
        and canonical_coolant_name(coolant) == canonical_coolant_name(fuel_name)
    )


def _apply_split_injector_pressure_model(args, parser) -> None:
    """Resolve authoritative fuel/oxidizer injector dP fractions for this run."""
    if args.fuel_injector_dp_fraction is None:
        args.fuel_injector_dp_fraction = _DEFAULT_INJECTOR_DP_FRACTION
    if args.oxidizer_injector_dp_fraction is None:
        args.oxidizer_injector_dp_fraction = _DEFAULT_INJECTOR_DP_FRACTION
    if args.fuel_injector_dp_fraction <= 0.0:
        parser.error("--fuel-injector-dp-fraction must be positive")
    if args.oxidizer_injector_dp_fraction <= 0.0:
        parser.error("--oxidizer-injector-dp-fraction must be positive")

    args._fuel_injector_pressure_drop = args.fuel_injector_dp_fraction * args.pc
    args._oxidizer_injector_pressure_drop = (
        args.oxidizer_injector_dp_fraction * args.pc
    )
    args._regen_injector_pressure_drop = 0.0
    args._regen_fuel_injector_dp_fraction = None
    args._regen_pressure_boundary_source = "pc_boundary_no_fuel_coolant_handoff"
    if args.coolant_outlet_pressure is not None:
        args._regen_pressure_boundary_source = "user_supplied_coolant_outlet_pressure"
    elif _coolant_is_cycle_fuel(args.coolant, getattr(args, "_fuel_name", None)):
        args._regen_injector_pressure_drop = args._fuel_injector_pressure_drop
        args._regen_fuel_injector_dp_fraction = args.fuel_injector_dp_fraction
        args._regen_pressure_boundary_source = (
            "fuel_injector_dp_fraction_split_model"
        )


def _positive_if_supplied(args, parser, names, *, allow_zero=False) -> None:
    for name in names:
        value = getattr(args, name)
        if value is None:
            continue
        bad = value < 0.0 if allow_zero else value <= 0.0
        if bad:
            op = "nonnegative" if allow_zero else "positive"
            parser.error(f"--{name.replace('_', '-')} must be {op}")


def _validate_common_engine_args(args, parser) -> None:
    """Run the workflow-independent scalar validation before dispatch.

    Historically the MDO and requirements branches returned before the
    traditional runner reached its input checks.  That made the same bad input
    produce either an argparse error or an uncaught numerical exception solely
    according to workflow.  Keep the shared mathematical domains in
    :mod:`raosim.input_validation` and call them before every early return.
    """

    from raosim.input_validation import (
        InputValidationError,
        validate_engine_inputs,
    )

    is_direct_mdo = bool(
        getattr(args, "engine_mdo", False)
        or getattr(args, "engine_mdo_optimize", False)
    )
    ambient = getattr(args, "engine_mdo_ambient", None)
    if is_direct_mdo and ambient is None:
        ambient = 101325.0
    altitude = None
    if getattr(args, "requirements", False):
        condition = str(getattr(args, "thrust_condition", "")).strip().lower()
        if condition.startswith("altitude"):
            try:
                parsed_condition = _parse_thrust_condition(condition)
            except ValueError as exc:
                parser.error(str(exc))
            altitude = float(parsed_condition[1])
    try:
        validate_engine_inputs(
            chamber_pressure=args.pc,
            expansion_ratio=args.epsilon,
            thrust=args.target_thrust,
            ambient_pressure=ambient if is_direct_mdo else None,
            ambient_pressure_ratio=args.pa_over_p0,
            altitude=altitude,
            mixture_ratio=args.mixture_ratio,
            burn_duration=args.burn_time,
            flight_duration=getattr(args, "flight_duration", None),
            qualification_duration=getattr(
                args, "qualification_duration", None
            ),
            isp_floor=getattr(args, "isp_min", None),
            envelope_diameter_max=getattr(
                args, "envelope_diameter_max", None
            ),
            envelope_length_max=getattr(args, "envelope_length_max", None),
            mass_max=getattr(args, "burnout_mass_max", None),
            film_fraction=getattr(args, "film_frac", None),
            injector_drop_fractions=(
                getattr(args, "fuel_injector_dp_fraction", None),
                getattr(args, "oxidizer_injector_dp_fraction", None),
            ),
            positive_dimensions=(
                ("channel_width", getattr(args, "channel_width", None)),
                ("channel_height", getattr(args, "channel_height", None)),
                ("t_wall", getattr(args, "t_wall", None)),
                ("film_slot_height", getattr(args, "film_slot_height", None)),
                ("pump_rpm", getattr(args, "pump_rpm", None)),
                (
                    "mdo_pc_search_min_pa",
                    getattr(args, "mdo_pc_search_min_pa", None),
                ),
                (
                    "mdo_pc_search_max_pa",
                    getattr(args, "mdo_pc_search_max_pa", None),
                ),
            ),
            reusable_cycles=getattr(args, "reusable_cycles", None),
        )
    except InputValidationError as exc:
        parser.error(str(exc))


def _validate_pump_args(args, parser) -> None:
    if args.pump_visualize:
        args.electric_pump = True
    if args.electric_pump and args.injector != "pintle":
        parser.error("--electric-pump requires --injector pintle")
    if (args.electric_pump and args.pump_cad != "none"
            and args.pump_cad_format in ("step", "both")):
        from raosim.pump_cad_brep import cadquery_available

        if not cadquery_available():
            parser.error(
                "--pump-cad-format step/both requires CadQuery/OpenCascade "
                "for the true B-rep pump path (pip install cadquery); the "
                "old faceted pseudo-STEP was removed - use "
                "--pump-cad-format stl for the mesh package"
            )
    feed_nonnegative = [
        "fuel_supply_pressure", "oxidizer_supply_pressure",
        "fuel_line_loss", "oxidizer_line_loss",
        "fuel_line_loss_fraction", "oxidizer_line_loss_fraction",
        "fuel_manifold_loss", "oxidizer_manifold_loss",
        "fuel_manifold_loss_fraction", "oxidizer_manifold_loss_fraction",
        "fuel_control_margin", "oxidizer_control_margin",
        "fuel_control_margin_fraction", "oxidizer_control_margin_fraction",
        "fuel_tank_pressure", "oxidizer_tank_pressure",
        "fuel_npsh_required", "oxidizer_npsh_required",
    ]
    _positive_if_supplied(args, parser, feed_nonnegative, allow_zero=True)
    _positive_if_supplied(args, parser, [
        "fuel_flow_capacity", "oxidizer_flow_capacity",
        "pump_rpm", "pump_max_rpm", "burn_time", "motor_voltage",
        "motor_power_density", "inverter_power_density", "battery_energy_density",
        "battery_power_density", "battery_structural_margin",
        "vehicle_mass", "pump_head_coefficient", "pump_flow_coefficient",
        "pump_tip_speed_limit", "pump_max_head_per_stage",
    ])
    _positive_if_supplied(args, parser, [
        "motor_max_power", "motor_max_current", "motor_torque_limit",
        "motor_heat_rejection_limit", "battery_voltage",
        "battery_max_current",
    ])
    for name in (
        "pump_efficiency_fuel", "pump_efficiency_oxidizer",
        "motor_efficiency", "inverter_efficiency",
        "battery_discharge_efficiency",
    ):
        value = getattr(args, name)
        if value is None:
            continue
        if not (0.0 < value <= 1.0):
            parser.error(f"--{name.replace('_', '-')} must be in (0, 1]")
    if args.battery_max_mass_fraction is not None:
        if not (0.0 < args.battery_max_mass_fraction <= 1.0):
            parser.error("--battery-max-mass-fraction must be in (0, 1]")


def _feed_system_spec_from_args(args):
    from raosim.injector import FeedLineSpec, FeedSystemSpec

    def line(role: str) -> FeedLineSpec:
        return FeedLineSpec(
            supply_pressure=getattr(args, f"{role}_supply_pressure"),
            flow_capacity=getattr(args, f"{role}_flow_capacity"),
            line_loss=getattr(args, f"{role}_line_loss"),
            line_loss_fraction=getattr(args, f"{role}_line_loss_fraction"),
            manifold_loss=getattr(args, f"{role}_manifold_loss"),
            manifold_loss_fraction=getattr(args, f"{role}_manifold_loss_fraction"),
            control_margin=getattr(args, f"{role}_control_margin"),
            control_margin_fraction=getattr(args, f"{role}_control_margin_fraction"),
            tank_pressure=getattr(args, f"{role}_tank_pressure"),
            npsh_required=getattr(args, f"{role}_npsh_required"),
            pump_efficiency=(
                getattr(args, f"pump_efficiency_{role}")
                if getattr(args, f"pump_efficiency_{role}") is not None
                else 0.7
            ),
        )

    return FeedSystemSpec(
        architecture=args.feed_architecture,
        fuel=line("fuel"),
        oxidizer=line("oxidizer"),
    )


def _pump_spec_from_args(args):
    from raosim.pumps import BatterySpec, ElectricDriveSpec, PumpSizingSpec

    return PumpSizingSpec(
        drive=ElectricDriveSpec(
            motor_efficiency=args.motor_efficiency,
            inverter_efficiency=args.inverter_efficiency,
            voltage=args.motor_voltage,
            rpm=args.pump_rpm,
            max_rpm=args.pump_max_rpm,
            max_motor_power=args.motor_max_power,
            max_current=args.motor_max_current,
            motor_power_density=args.motor_power_density,
            inverter_power_density=args.inverter_power_density,
            torque_limit=args.motor_torque_limit,
            heat_rejection_limit=args.motor_heat_rejection_limit,
        ),
        battery=BatterySpec(
            energy_density=args.battery_energy_density,
            power_density=args.battery_power_density,
            discharge_efficiency=args.battery_discharge_efficiency,
            structural_margin=args.battery_structural_margin,
            voltage=args.battery_voltage,
            max_current=args.battery_max_current,
            max_mass_fraction=args.battery_max_mass_fraction,
            vehicle_mass=args.vehicle_mass,
        ),
        burn_time=args.burn_time,
        pump_efficiency={
            "fuel": args.pump_efficiency_fuel,
            "oxidizer": args.pump_efficiency_oxidizer,
        },
        head_coefficient=args.pump_head_coefficient,
        flow_coefficient=args.pump_flow_coefficient,
        material_tip_speed_limit=args.pump_tip_speed_limit,
        max_head_per_stage=args.pump_max_head_per_stage,
    )


def _apply_wall_sizing_mode(args, parser, argv) -> None:
    """Resolve the user-facing wall-sizing mode into the existing switches."""
    wall_thickness_given = _argument_present(argv, "--wall-thickness")
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
    if getattr(args, "_complete_package_defaults", False):
        notes = ", ".join(args._complete_package_default_notes)
        print(dim(f"    complete-package defaults applied: {notes}"))

    _section("Propellant and operating point")
    args.propellant = _prompt_optional_str(
        "Propellant pair OX/FUEL", args.propellant
    )
    if args.propellant is None:
        args.oxidizer = _prompt_optional_str("Oxidizer name", args.oxidizer)
        args.fuel = _prompt_optional_str("Fuel name", args.fuel)
    else:
        if "/" not in args.propellant:
            args.oxidizer = _prompt_optional_str("Oxidizer name", args.oxidizer)
            args.fuel = _prompt_optional_str("Fuel name", args.fuel)
    args.thermo_mode = _prompt_choice(
        "Thermochemistry mode (constant-gamma / cea)",
        args.thermo_mode,
        {"constant-gamma", "cea"},
    )
    args.pc = _prompt("Chamber pressure Pc [Pa]", args.pc)
    args.mixture_ratio = _prompt_optional_float(
        "Mixture ratio O/F (blank = propellant nominal)",
        args.mixture_ratio,
    )
    args.target_thrust = _prompt_optional_float(
        "Target thrust [N] (blank = use Rt)", args.target_thrust
    )

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
    if _prompt_bool("Auto-size chamber shoulder fillet?", args.shoulder_sizing == "auto"):
        args.shoulder_sizing = "auto"
        args.shoulder_radius_factor = None
        args._shoulder_radius_source = "auto_pending"
        args.shoulder_fill_fraction = _prompt(
            "Shoulder fill fraction of max feasible", args.shoulder_fill_fraction
        )
    else:
        args.shoulder_sizing = "scalar"
        shoulder_default = (
            args.shoulder_radius_factor
            if args.shoulder_radius_factor is not None else 0.25
        )
        args.shoulder_radius_factor = _prompt(
            "Shoulder radius / Rt", shoulder_default
        )
        args._shoulder_radius_source = "user_supplied"
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
        "Render the resolved steady field (Mach, pressure, angle, temperature)?",
        bool(args.flowfield))
    if _prompt_bool("Animate the flow (MOC march + particle advection)?",
                    args.animate is not None):
        args.animate = _prompt(
            "  which? (march / particles / both)", args.animate or "both", str)

    _section("Pintle injector")
    use_pintle = _prompt_bool(
        "Generate the selected pintle injector package?",
        args.injector == "pintle",
    )
    args.injector = "pintle" if use_pintle else "none"
    if args.injector == "pintle":
        args.injector_architecture = _prompt_choice(
            "Injector architecture (fixed_discrete / son_continuous_movable)",
            args.injector_architecture,
            {"fixed_discrete", "son_continuous_movable"},
        )
        args.injector_sizing = _prompt_choice(
            "Injector sizing mode (auto / fixed / movable)",
            args.injector_sizing,
            {"auto", "fixed", "movable"},
        )
        args.pintle_radial_stream = _prompt_choice(
            "Radial/slotted stream (fuel / oxidizer)",
            args.pintle_radial_stream,
            {"fuel", "oxidizer"},
        )
        args.fuel_injector_dp_fraction = _prompt(
            "Fuel injector pressure drop ΔP/Pc",
            (args.fuel_injector_dp_fraction
             if args.fuel_injector_dp_fraction is not None
             else _DEFAULT_INJECTOR_DP_FRACTION),
        )
        args.oxidizer_injector_dp_fraction = _prompt(
            "Oxidizer injector pressure drop ΔP/Pc",
            (args.oxidizer_injector_dp_fraction
             if args.oxidizer_injector_dp_fraction is not None
             else _DEFAULT_INJECTOR_DP_FRACTION),
        )
        args.fuel_discharge_coefficient = _prompt(
            "Fuel discharge coefficient Cd", args.fuel_discharge_coefficient
        )
        args.oxidizer_discharge_coefficient = _prompt(
            "Oxidizer discharge coefficient Cd",
            args.oxidizer_discharge_coefficient,
        )
        args.pintle_diameter = _prompt_optional_float(
            "Pintle diameter [m] (blank = auto)", args.pintle_diameter
        )
        args.pintle_slot_count = _prompt(
            "Pintle radial opening count", args.pintle_slot_count, int
        )
        args.pintle_radial_exit = _prompt_choice(
            "Radial exit style (holes / slots / continuous_radial_gap)",
            args.pintle_radial_exit,
            {"holes", "slots", "continuous_radial_gap"},
        )
        if args.pintle_radial_exit == "continuous_radial_gap":
            args.movable_post_diameter = _prompt_optional_float(
                "Movable post diameter D_post [m]",
                args.movable_post_diameter,
            )
            args.movable_post_thickness = _prompt_optional_float(
                "Movable post thickness t_post [m]",
                args.movable_post_thickness,
            )
            args.movable_center_gap_diameter = _prompt_optional_float(
                "Centre-gap diameter D_cg [m]",
                args.movable_center_gap_diameter,
            )
            args.movable_pintle_rod_diameter = _prompt_optional_float(
                "Centre rod diameter D_pr [m]",
                args.movable_pintle_rod_diameter,
            )
        args.pintle_slot_aspect_ratio = _prompt(
            "Pintle slot aspect ratio h/w", args.pintle_slot_aspect_ratio
        )
        args.pintle_deflector_angle = _prompt(
            "Pintle deflector angle [deg]", args.pintle_deflector_angle
        )
        args.pintle_target_momentum_ratio = _prompt_optional_float(
            "Target total momentum ratio (blank = no gate)",
            args.pintle_target_momentum_ratio,
        )
        args.pintle_impingement_distance = _prompt_optional_float(
            "Impingement distance [m] (blank = auto)",
            args.pintle_impingement_distance,
        )
        args.injector_min_feature = _prompt(
            "Minimum injector feature [m]", args.injector_min_feature
        )
        if args.injector_sizing in ("fixed", "movable") or _prompt_bool(
            "Edit fixed passage overrides?", False
        ):
            args.pintle_annulus_gap = _prompt_optional_float(
                "Fixed annulus gap [m]", args.pintle_annulus_gap
            )
            if args.pintle_radial_exit == "continuous_radial_gap":
                if args.injector_sizing == "fixed":
                    args.movable_commanded_opening = _prompt_optional_float(
                        "Fixed mechanical opening L_open [m]",
                        args.movable_commanded_opening,
                    )
                args.movable_maximum_opening = _prompt_optional_float(
                    "Physical open stop [m] (blank = derived)",
                    args.movable_maximum_opening,
                )
            elif args.pintle_radial_exit == "holes":
                args.pintle_hole_diameter = _prompt_optional_float(
                    "Fixed radial hole diameter [m]",
                    args.pintle_hole_diameter,
                )
                args.pintle_hole_length = _prompt_optional_float(
                    "Fixed radial hole length [m]", args.pintle_hole_length
                )
            else:
                args.pintle_slot_width = _prompt_optional_float(
                    "Fixed slot width [m]", args.pintle_slot_width
                )
                args.pintle_slot_height = _prompt_optional_float(
                    "Fixed slot height [m]", args.pintle_slot_height
                )
                args.pintle_slot_depth = _prompt_optional_float(
                    "Fixed slot depth [m]", args.pintle_slot_depth
                )
            args.pintle_tip_radius = _prompt_optional_float(
                "Pintle tip radius [m]", args.pintle_tip_radius
            )
            args.pintle_body_length = _prompt_optional_float(
                "Pintle body length [m]", args.pintle_body_length
            )
        if _prompt_bool("Edit injector face, ports, and bolt pattern?", False):
            args.injector_face_od = _prompt_optional_float(
                "Injector face outer diameter [m]", args.injector_face_od
            )
            args.injector_face_thickness = _prompt_optional_float(
                "Injector face thickness [m]", args.injector_face_thickness
            )
            args.bolt_count = _prompt("Bolt count", args.bolt_count or 8, int)
            args.bolt_circle = _prompt_optional_float(
                "Bolt circle diameter [m]", args.bolt_circle
            )
            args.bolt_hole = _prompt_optional_float(
                "Bolt hole diameter [m]", args.bolt_hole
            )
            args.bolt_diameter = _prompt_optional_float(
                "Bolt tensile diameter [m]", args.bolt_diameter
            )
            args.fuel_inlet_count = _prompt(
                "Fuel inlet count", args.fuel_inlet_count, int
            )
            args.fuel_inlet_diameter = _prompt_optional_float(
                "Fuel inlet diameter [m]", args.fuel_inlet_diameter
            )
            args.oxidizer_inlet_count = _prompt(
                "Oxidizer inlet count", args.oxidizer_inlet_count, int
            )
            args.oxidizer_inlet_diameter = _prompt_optional_float(
                "Oxidizer inlet diameter [m]", args.oxidizer_inlet_diameter
            )
            args.fuel_manifold_width = _prompt_optional_float(
                "Fuel manifold width [m]", args.fuel_manifold_width
            )
            args.fuel_manifold_depth = _prompt_optional_float(
                "Fuel manifold depth [m]", args.fuel_manifold_depth
            )
            args.oxidizer_manifold_width = _prompt_optional_float(
                "Oxidizer manifold width [m]", args.oxidizer_manifold_width
            )
            args.oxidizer_manifold_depth = _prompt_optional_float(
                "Oxidizer manifold depth [m]", args.oxidizer_manifold_depth
            )
        args.injector_cad = _prompt_choice(
            "Pintle CAD package (auto / none / reference / parts / machined)",
            args.injector_cad,
            {"auto", "none", "reference", "parts", "machined", "step"},
        )
        args.injector_cad_format = _prompt_choice(
            "Pintle CAD format (step / stl / dxf)",
            args.injector_cad_format,
            {"step", "stl", "dxf"},
        )
        args.pintle_sleeve = _prompt_bool(
            "Include movable pintle sleeve body?", bool(args.pintle_sleeve)
        )

    _section("Feed system and electric pump")
    if args.injector == "pintle":
        args.electric_pump = _prompt_bool(
            "Size the selected electric pump package?", bool(args.electric_pump)
        )
        args.feed_architecture = _prompt_choice(
            "Feed architecture (pump_fed / pressure_fed)",
            args.feed_architecture,
            {"pump_fed", "pressure_fed"},
        )
        args.fuel_tank_pressure = _prompt_optional_float(
            "Fuel tank / pump inlet pressure [Pa]", args.fuel_tank_pressure
        )
        args.oxidizer_tank_pressure = _prompt_optional_float(
            "Oxidizer tank / pump inlet pressure [Pa]",
            args.oxidizer_tank_pressure,
        )
        args.fuel_supply_pressure = _prompt_optional_float(
            "Available fuel pump outlet pressure [Pa]",
            args.fuel_supply_pressure,
        )
        args.oxidizer_supply_pressure = _prompt_optional_float(
            "Available oxidizer pump outlet pressure [Pa]",
            args.oxidizer_supply_pressure,
        )
        args.fuel_flow_capacity = _prompt_optional_float(
            "Available fuel flow capacity [kg/s]", args.fuel_flow_capacity
        )
        args.oxidizer_flow_capacity = _prompt_optional_float(
            "Available oxidizer flow capacity [kg/s]",
            args.oxidizer_flow_capacity,
        )
        args.fuel_line_loss_fraction = _prompt(
            "Fuel line loss fraction of Pc", args.fuel_line_loss_fraction
        )
        args.oxidizer_line_loss_fraction = _prompt(
            "Oxidizer line loss fraction of Pc",
            args.oxidizer_line_loss_fraction,
        )
        args.fuel_control_margin_fraction = _prompt(
            "Fuel control margin fraction of Pc",
            args.fuel_control_margin_fraction,
        )
        args.oxidizer_control_margin_fraction = _prompt(
            "Oxidizer control margin fraction of Pc",
            args.oxidizer_control_margin_fraction,
        )
        if args.electric_pump:
            args.pump_rpm = _prompt_optional_float(
                "Pump speed [rpm] (blank = auto)", args.pump_rpm
            )
            args.pump_max_rpm = _prompt(
                "Pump maximum speed [rpm]", args.pump_max_rpm
            )
            args.burn_time = _prompt(
                "Burn time for battery sizing [s]", args.burn_time
            )
            args.motor_voltage = _prompt_optional_float(
                "Motor DC bus voltage [V] (blank = auto)", args.motor_voltage
            )
            args.motor_efficiency = _prompt(
                "Motor efficiency", args.motor_efficiency
            )
            args.inverter_efficiency = _prompt(
                "Inverter efficiency", args.inverter_efficiency
            )
            args.battery_energy_density = _prompt(
                "Battery usable energy density [J/kg]",
                args.battery_energy_density,
            )
            args.battery_power_density = _prompt(
                "Battery pulse power density [W/kg]",
                args.battery_power_density,
            )
            args.pump_head_coefficient = _prompt(
                "Pump head coefficient psi", args.pump_head_coefficient
            )
            args.pump_flow_coefficient = _prompt(
                "Pump flow coefficient phi", args.pump_flow_coefficient
            )
            args.pump_tip_speed_limit = _prompt(
                "Pump tip-speed limit [m/s]", args.pump_tip_speed_limit
            )
            args.pump_cad = _prompt_choice(
                "Pump CAD package (auto / none / reference / parts)",
                args.pump_cad,
                {"auto", "none", "reference", "parts"},
            )
            args.pump_cad_format = _prompt_choice(
                "Pump CAD format (stl / step B-rep / both)",
                args.pump_cad_format,
                {"stl", "step", "both"},
            )
            args.pump_visualize = _prompt_bool(
                "Save pump particle GIF?", bool(args.pump_visualize)
            )
    else:
        args.electric_pump = False

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


class _TrustedTopSolution:
    """Small compatibility adapter for the deterministic Rao/TOP path.

    The CLI historically assumed every contour came from ``RaoSolution``.
    Keeping a common read-only result surface lets the trusted chart path be
    the default without fabricating or running an optimization problem.
    """

    def __init__(self, contour: dict, cf_ideal: float):
        from types import SimpleNamespace

        self._contour = contour
        self.theta_N = math.radians(float(contour["theta_n"]))
        self.theta_E = math.radians(float(contour["theta_e"]))
        self.thrust_coefficient = float(cf_ideal)
        self.residuals = SimpleNamespace(
            max_scaled=0.0,
            rms_scaled=0.0,
            mass_residual_rel=0.0,
            length_residual_rel=0.0,
            wall_tangency_rms=0.0,
            characteristic_crossings=0,
        )
        extrapolated = bool(contour.get("rao_chart_extrapolated", False))
        self.reliability = (
            rv.ContourReliability.GEOMETRIC_APPROXIMATION
            if extrapolated else rv.ContourReliability.BENCHMARK_VALIDATED
        )
        self.converged = True
        self.warnings = list(contour.get("warnings", []))
        self.construction_diagnostics = {
            "contour_method": "bezier",
            "rao_chart_domain": contour.get("rao_chart_domain"),
            "rao_chart_extrapolated": extrapolated,
            "design_angles": {
                "theta_N_source": contour.get("angle_source", {}).get(
                    "theta_n", "rao_top_chart"
                ),
                "theta_E_source": contour.get("angle_source", {}).get(
                    "theta_e", "rao_top_chart"
                ),
            },
            "postprocessed": False,
            "moc_compatibility_preserved": True,
            "wall_tangency_rms": 0.0,
            "boundary_min": 0.0,
            "thrust_sanity": {
                "applicable": True,
                "passes": True,
                "gate_basis": "quasi_1d_attached_flow",
                "cf_surface": float(cf_ideal),
                "cf_ideal": float(cf_ideal),
                "cf_rel_error": 0.0,
            },
            "net_report": {},
        }

    def to_contour_dict(self, **_kwargs) -> dict:
        return dict(self._contour)


def _solve(args):
    if args.contour_method == "bezier":
        from raosim.nozzle_geometry import bell_nozzle_contour

        contour = bell_nozzle_contour(
            Rt=args.rt,
            epsilon=args.epsilon,
            length_pct=args.length_pct,
            gamma=args.gamma,
            pa_over_p0=args.pa_over_p0,
            Ru_factor=args.ru_factor,
            Rd_factor=args.rd_factor,
            convergent_half_angle_deg=args.convergent_angle,
        )
        if args._performance.frozen_flow is not None:
            from raosim.validation import add_contour_reliability_metadata

            add_contour_reliability_metadata(
                contour,
                "bezier",
                args.gamma,
                frozen_expansion=args._performance.frozen_flow,
            )
        return _TrustedTopSolution(contour, args._performance.Cf_ideal)

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
        physics_weight=1.0,
        thetaN_guess_deg=args.theta_b_guess,
    )
    return rv.solve_rao_bvp(cfg)


def _print_injector_panel(inj) -> None:
    """Console panel for a sized pintle injector."""
    fs = inj.streams["fuel"]
    os = inj.streams["oxidizer"]
    radial_style = inj.slots.geometry
    print(f"    {green('streams:'):<16}radial={inj.radial_stream} "
          f"({radial_style if inj.radial_stream == inj.slots.role else 'annulus'})  "
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
    if radial_style == "holes":
        radial_description = (
            f"holes Ø{sd['hole_diameter']*1e3:.3f} mm ×{inj.slot_count} "
            f"L={sd['hole_length']*1e3:.3f} mm"
        )
    elif radial_style == "slots":
        radial_description = (
            f"slots {sd['slot_width']*1e3:.3f}×"
            f"{sd['slot_height']*1e3:.3f} mm ×{inj.slot_count}"
        )
    else:
        radial_description = (
            f"Son L_open={sd['opening_distance']*1e3:.4f} mm  "
            f"A_tip={sd['tip_minimum_area']*1e6:.3f} mm²  "
            f"A_cg={sd['center_gap_area']*1e6:.3f} mm²"
        )
    if radial_style == "continuous_radial_gap":
        print(
            f"    {green('geometry:'):<16}"
            f"annulus gap={ad['gap']*1e3:.3f} mm   {radial_description}   "
            "continuous 360° sheet (no discrete web)"
        )
        act = inj.actuation
        if act is not None:
            force_margin = (
                f"{act.actuator_force_margin:.2f}"
                if act.actuator_force_margin is not None else "—"
            )
            print(
                f"    {green('actuation:'):<16}"
                f"stroke={act.opening_fraction:.3f} of open stop   "
                f"Cd={act.discharge_coefficient:.3f} "
                f"({act.discharge_coefficient_model})   "
                f"static force margin={force_margin}"
            )
    else:
        print(
            f"    {green('geometry:'):<16}"
            f"annulus gap={ad['gap']*1e3:.3f} mm   "
            f"{radial_description}   web={inj.minimum_web*1e3:.3f} mm   "
            f"blockage={inj.blockage_factor:.2f}"
        )
    wall = (f"{inj.spray_wall_axial_distance*1e3:.0f} mm"
            if inj.spray_wall_axial_distance == inj.spray_wall_axial_distance
            and inj.spray_wall_axial_distance != float("inf") else "—")
    print(f"    {green('spray:'):<16}TMR={inj.total_momentum_ratio:.2f}   "
          f"half-angle={inj.spray_half_angle_deg:.0f}°   "
          f"wall@{wall}   opening/gap="
          f"{inj.slot_to_annulus_width_ratio:.2f}")
    at = getattr(inj, "atomization", None)
    if at is not None:
        applicable = [s for s in at.streams.values() if s.applicable]
        if at.limiting_role is None or not applicable:
            print(f"    {green('atomization:'):<16}"
                  "not applicable: no liquid droplet stream; "
                  "eta_vaporization unavailable")
        else:
            lim = at.streams[at.limiting_role]
            print(f"    {green('atomization:'):<16}"
                  f"SMD {lim.sauter_mean_diameter*1e6:.0f} µm "
                  f"({at.limiting_role}; {len(applicable)} liquid stream"
                  f"{'s' if len(applicable) != 1 else ''})   "
                  f"L_dev {at.combustion_length*1e3:.0f} mm   "
                  f"margin {at.development_margin:.2f}   "
                  f"η_vaporization≈{at.eta_vaporization:.2f} "
                  + dim("(eta_mixing/eta_combustion/eta_c* unavailable)"))
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


def _print_pump_panel(pump) -> None:
    """Console panel for electric pump sizing."""
    total_shaft = sum(
        (ln.shaft_power or 0.0) for ln in pump.lines.values()
    )
    total_elec = pump.battery.electric_power
    print(f"    {green('power:'):<16}shaft {total_shaft/1000:.2f} kW   "
          f"electric {total_elec/1000:.2f} kW   "
          f"battery {pump.battery.mass:.2f} kg ({pump.battery.limiting})")
    print(f"    {green('battery:'):<16}{pump.battery.current:.0f} A   "
          f"energy {pump.battery.energy_required/1000:.1f} kJ   "
          f"heat {pump.battery.heat:.0f} W")
    for role, ln in pump.lines.items():
        if ln.impeller is None or ln.drive is None:
            print(f"    {green(role + ':'):<16}"
                  "tank pressure missing; head/power/geometry not sized")
            continue
        imp = ln.impeller
        ind = ln.inducer
        dif = ln.diffuser_volute
        print(f"    {green(role + ':'):<16}"
              f"H={ln.head:.0f} m  rise={ln.pressure_rise/1e5:.1f} bar  "
              f"Pshaft={ln.shaft_power/1000:.2f} kW  "
              f"T={ln.drive.torque:.3g} N m")
        print(f"    {dim(''):<16}"
              f"rpm={ln.drive.rpm:.0f}  V={ln.drive.voltage:.0f}  "
              f"ηp={ln.efficiency:.2f} ({ln.efficiency_source})")
        print(f"    {dim(''):<16}"
              f"D2={imp.impeller_diameter*1e3:.1f} mm  "
              f"b2={imp.outlet_width*1e3:.2f} mm  "
              f"U2={imp.tip_speed:.0f} m/s  Ns={imp.specific_speed:.2f}  "
              f"{dif.selection if dif else 'diffuser'}")
        if getattr(ln, "architecture", None) is not None:
            arch = ln.architecture
            print(f"    {dim(''):<16}"
                  f"architecture={arch.primary_type}  "
                  f"{arch.stage_mode}  {arch.suction_assist}")
        if ln.hydraulic_meanline is not None:
            ml = ln.hydraulic_meanline
            tri = ml.velocity_triangle
            re = ml.losses.reynolds_number
            re_text = "unknown" if re is None else f"{re:.2g}"
            print(f"    {dim(''):<16}"
                  f"slip={tri.slip_factor:.2f}  "
                  f"Euler margin={tri.euler_head_margin:.0f} m/stage  "
                  f"loss={ml.losses.total_loss_head:.0f} m/stage  "
                  f"Re={re_text}")
        if ind is not None:
            ss = "unknown" if ind.suction_specific_speed is None else f"{ind.suction_specific_speed:.2f}"
            print(f"    {dim(''):<16}"
                  f"inducer D={ind.diameter*1e3:.1f} mm  "
                  f"NSS={ss}  NPSH margin="
                  f"{'unknown' if ind.npsh_margin is None else f'{ind.npsh_margin/1e5:+.2f} bar'}")
    n_pass = sum(g.status == "pass" for g in pump.feasibility.gates)
    n_warn = sum(g.status == "warn" for g in pump.feasibility.gates)
    n_fail = sum(g.status == "fail" for g in pump.feasibility.gates)
    print(f"    {green('gates:'):<16}{green(str(n_pass)+' pass')}  "
          f"{yellow(str(n_warn)+' warn')}  "
          f"{red(str(n_fail)+' fail') if n_fail else dim('0 fail')}")
    for g in pump.feasibility.gates:
        if g.status == "fail":
            print(red(f"      ✗ {g.name}: {g.detail}"))
        elif g.status == "warn":
            print(yellow(f"      ● {g.name}: {g.detail}"))
    if pump.feasible:
        print(f"    {green('verdict:'):<16}{green('✓ screening pass')}")
    else:
        print(f"    {green('verdict:'):<16}{red('✗ electric pump screening failed')}")
    for suggestion in pump.feasibility.suggestions[:3]:
        print(yellow(f"    suggestion: {suggestion}") if not pump.feasible
              else dim(f"    note: {suggestion}"))


def _cooling_summary_payload(cooling_result, args) -> dict:
    """Serialize the final cooling iterate without stale pre-loop values."""

    passage_factor = cooling_result.get("passage_length_factor", 1.0)
    return {
        "peak_wall_T_K": cooling_result["peak_gas_side_wall_temperature"],
        "cooling_margin": cooling_result["cooling_margin"],
        "coolant_outlet_T_K": cooling_result["coolant_outlet_temperature"],
        "coolant_inlet_T_K": cooling_result.get(
            "coolant_inlet_temperature", args.coolant_inlet_temperature
        ),
        "coolant_mass_flow_kg_s": args.coolant_mdot,
        "coolant_chemistry_margin": cooling_result.get(
            "coolant_chemistry_margin"
        ),
        "coolant_chemistry_status": cooling_result.get(
            "coolant_chemistry_status"
        ),
        "pressure_drop_bar": cooling_result["coolant_pressure_drop"] / 1e5,
        "channel_roughness_m": cooling_result.get("channel_roughness"),
        "pressure_drop_correlation": cooling_result.get(
            "pressure_drop_correlation"
        ),
        "peak_channel_velocity_m_s": cooling_result.get(
            "peak_channel_velocity"
        ),
        "gas_radiation_status": cooling_result.get("gas_radiation_status"),
        "coolant_property_backend": cooling_result.get(
            "coolant_property_backend"
        ),
        "coolant_property_status": cooling_result.get(
            "coolant_property_status"
        ),
        "hydraulic_network_status": cooling_result.get(
            "hydraulic_network_status"
        ),
        "channel_friction_pressure_drop_bar": (
            cooling_result.get("channel_friction_pressure_drop", 0.0) / 1e5
        ),
        "boiling_chf_status": cooling_result.get("boiling_chf_status"),
        "curvature_correlation_status": cooling_result.get(
            "curvature_correlation_status"
        ),
        "helix_turns": cooling_result.get("helix_turns", 0.0),
        "passage_length_factor": passage_factor,
        "coolant_inlet_pressure_Pa": cooling_result.get(
            "coolant_inlet_pressure"
        ),
        "coolant_outlet_pressure_Pa": cooling_result.get(
            "coolant_outlet_pressure"
        ),
        "coolant_pressure_boundary_source": cooling_result.get(
            "coolant_pressure_boundary_source"
        ),
        "outer_loop_state": "final_spray_cstar_regen_iterate",
    }


_MODE_MENU = """
==============================================================================
  LREKit — liquid rocket engine design
==============================================================================
  Choose a workflow:

    1) Traditional solver      nozzle contour + chamber/injector/pump sizing,
                               reports + CAD export (the classic LREKit run)

    2) Whole-engine MDO        ONE coupled, differentiable evaluation of the
       (single point)          nozzle + regen cooling + pintle injector +
                               electric pump feed at YOUR design point, with
                               every constraint margin reported

    3) Whole-engine MDO        the optimiser: you give requirements + an Isp
       (optimise)              target, it SOLVES for the design (Pc, eps,
                               injector dP, pintle dia, pump rpm, cooling
                               channels, film fraction) at minimum mass
==============================================================================
"""


def _interactive_engine_mdo(args, *, optimise: bool) -> None:
    """Prompt for the parameters the MDO workflows actually use.

    Deliberately short: mission requirements first, then the cooling choice
    (pure regen vs fuel-film), then the mode-specific settings.
    """
    print("\n-- mission requirements ---------------------------------------")
    args.target_thrust = _prompt("Target thrust [N]",
                                 args.target_thrust or 13000.0)
    args.mixture_ratio = _prompt("Mixture ratio O/F",
                                 args.mixture_ratio or 2.27)
    args.burn_time = _prompt("Burn time [s]", args.burn_time or 120.0)
    args.engine_mdo_ambient = _prompt(
        "Ambient pressure [Pa]  (101325 = sea level, ~1000 = high altitude)",
        args.engine_mdo_ambient if args.engine_mdo_ambient is not None
        else 101325.0)

    print("\n-- cooling ----------------------------------------------------")
    cooling = _prompt_choice(
        "Cooling: 'regen' (regenerative only) or 'film' (regen + fuel film)",
        "film" if optimise else "regen", ("regen", "film"))
    if cooling == "regen":
        args.film_frac = 0.0
        print("    pure regenerative cooling (film fraction = 0)")
        if optimise:
            print("    NOTE: with RP-1 the wall is coolant-enthalpy-limited, so "
                  "pure regen\n          usually violates the SP-8087 coking "
                  "limit — expect an infeasible\n          solve unless you "
                  "lower Pc a lot.  'film' is the physical fix.")
    elif not optimise:
        args.film_frac = _prompt("Fuel film fraction (0-0.3)",
                                 args.film_frac if args.film_frac is not None
                                 else 0.05)
        args.film_slot_height = _prompt(
            "Film-injector slot height [m]",
            args.film_slot_height if args.film_slot_height is not None
            else 2.0e-3)

    if optimise:
        print("\n-- optimiser --------------------------------------------------")
        sweep = _prompt_bool("Trace the mass-Isp Pareto frontier "
                             "(no = single min-mass solve)", False)
        if sweep:
            lo = _prompt("  Isp floor: from [s]", 190.0)
            hi = _prompt("  Isp floor: to [s]", 210.0)
            n = int(_prompt("  number of points", 5))
            args.isp_sweep = f"{lo},{hi},{n}"
        else:
            args.isp_min = _prompt("Minimum Isp [s] (the epsilon-constraint)",
                                   args.isp_min or 200.0)
    else:
        print("\n-- design point -----------------------------------------------")
        args.pc = _prompt("Chamber pressure Pc [Pa]", args.pc or 3.0e6)
        args.epsilon = _prompt("Expansion ratio eps", args.epsilon or 8.0)
        args.fuel_injector_dp_fraction = _prompt(
            "Fuel injector dP / Pc", args.fuel_injector_dp_fraction or 0.20)
        args.oxidizer_injector_dp_fraction = _prompt(
            "Oxidizer injector dP / Pc",
            args.oxidizer_injector_dp_fraction or 0.20)
        args.channel_height = _prompt("Cooling channel height [m]",
                                      args.channel_height or 3.0e-3)
    args.design_margins = _prompt_bool(
        "Apply SP-8087/Mirzamoghadam design margins (heat flux x1.10, channel "
        "flow x0.90, film capacity x2) instead of nominal", False)
    args.engine_mdo_couple_cstar = _prompt_bool(
        "Enable the spray->eta_c* feedback edge (screening correlation; "
        "default off = frozen + ablation)", False)


def _select_mode_interactively(args) -> str:
    """Bare-run front door: pick the workflow, then gather its parameters.

    Returns 'traditional' | 'mdo' | 'mdo_optimize'.
    """
    print(_MODE_MENU)
    choice = _prompt_choice("Workflow", "1", ("1", "2", "3"))
    if choice == "2":
        _interactive_engine_mdo(args, optimise=False)
        return "mdo"
    if choice == "3":
        _interactive_engine_mdo(args, optimise=True)
        return "mdo_optimize"
    return "traditional"


def _mdo_contour_request_error(args) -> str | None:
    """Validate the host-only exact-contour selector for MDO workflows."""

    if args.contour_method != "rao-bvp":
        return None
    if not getattr(args, "mdo_export", False):
        return (
            "--contour-method rao-bvp is a post-analysis selector for MDO and "
            "requires --mdo-export; the differentiable MDO core remains on its "
            "fixed-topology Rao/TOP chart wall."
        )
    if args.cad != "none":
        return (
            "--contour-method rao-bvp is a preliminary numerical "
            "post-analysis and cannot emit manufacturing CAD; use --cad none "
            "or retain the default Bezier authoritative handoff."
        )
    return None


def _mdo_authoritative_contour_handoff(
    args,
) -> tuple[str, dict[str, object] | None]:
    """Resolve the host contour method and traditional-CLI-equivalent options."""

    if args.contour_method != "rao-bvp":
        return "bezier", None
    return "rao_variational_moc", {
        "n_control": int(args.n_control),
        "n_kernel": int(args.n_kernel),
        "max_nfev": int(args.max_nfev),
        "evaluate_moc": True,
        "theta_n_guess_deg": float(args.theta_b_guess),
        "starting_line_method": "kliegel_levine",
        "solver_backend": str(args.backend),
        "wall_method": "bde",
        "kernel_d_fraction_max": 0.7,
        "physics_weight": 1.0,
    }


def _resolve_mdo_of_intent(args, *, optimization_capable: bool):
    """Resolve CLI mixture-ratio intent without overloading ``None``."""

    from raosim.requirements import MixtureRatioMode

    explicit_mode = getattr(args, "mdo_of_mode", None)
    mode = MixtureRatioMode(
        explicit_mode
        if explicit_mode is not None
        else (
            MixtureRatioMode.PINNED.value
            if args.mixture_ratio is not None
            else MixtureRatioMode.NOMINAL.value
        )
    )
    ratio = args.mixture_ratio
    if mode is MixtureRatioMode.NOMINAL and ratio is not None:
        raise ValueError(
            "--mdo-of-mode nominal cannot be combined with --mixture-ratio; "
            "use --mdo-of-mode pinned"
        )
    if mode is MixtureRatioMode.PINNED and ratio is None:
        raise ValueError(
            "--mdo-of-mode pinned requires --mixture-ratio"
        )
    if mode is MixtureRatioMode.OPTIMIZE:
        if not optimization_capable:
            raise ValueError(
                "--mdo-of-mode optimize requires --engine-mdo-optimize or "
                "--requirements"
            )
        if getattr(args, "mdo_chamber_property_table", None) is None:
            raise ValueError(
                "--mdo-of-mode optimize requires "
                "--mdo-chamber-property-table"
            )
    return mode, ratio


def _mission_from_mdo_args(args, *, optimization_capable: bool):
    """Build one MissionSpec for direct evaluation and optimization CLIs."""

    from raosim.mdo.schema import MissionSpec
    from raosim.requirements import MixtureRatioMode

    mode, ratio = _resolve_mdo_of_intent(
        args, optimization_capable=optimization_capable
    )
    thrust = args.target_thrust if args.target_thrust is not None else 13.0e3
    overrides = {
        "of_is_pinned": mode is not MixtureRatioMode.OPTIMIZE,
    }
    if ratio is not None:
        overrides["OF"] = float(ratio)
    if args.pump_rpm is not None:
        overrides["pump_speed_rpm"] = float(args.pump_rpm)
    if getattr(args, "engine_mdo_ambient", None) is not None:
        overrides["Pa"] = float(args.engine_mdo_ambient)
    if getattr(args, "burn_time", None) is not None:
        overrides["burn_time"] = float(args.burn_time)
    if getattr(args, "design_margins", False):
        overrides.update(
            heat_flux_margin=1.10,
            channel_flow_margin=0.90,
            film_capacity_margin=2.0,
        )
    table = getattr(args, "mdo_chamber_property_table", None)
    if table is not None:
        overrides["cea_table_path"] = str(Path(table).expanduser())
    pc_min = getattr(args, "mdo_pc_search_min_pa", None)
    pc_max = getattr(args, "mdo_pc_search_max_pa", None)
    if pc_min is not None:
        overrides["chamber_pressure_search_min_pa"] = float(pc_min)
    if pc_max is not None:
        overrides["chamber_pressure_search_max_pa"] = float(pc_max)

    propellant = getattr(args, "mdo_propellant", None)
    if propellant:
        mission = MissionSpec.for_propellant(
            propellant, float(thrust), **overrides
        )
    else:
        mission = MissionSpec.for_thrust(float(thrust), **overrides)

    # ``--material``/``--jacket-material`` must reach the differentiable core,
    # not only the traditional pipeline.  Before this the MDO kept its flat
    # class defaults, so selecting GRCop-84 still optimized a NARloy-Z-class
    # liner.  The mapping is atomic: an unknown or incompletely specified alloy
    # raises here and the CLI reports invalid input rather than silently
    # optimizing an unattributed wall.
    liner_material = getattr(args, "material", None)
    if liner_material:
        mission = mission.with_materials(
            liner=liner_material,
            closeout=getattr(args, "jacket_material", None) or "Inconel 718",
        )
    return mission, mode


def _run_engine_mdo(args) -> int:
    """Whole-engine differentiable MDO evaluation (``raosim.mdo.engine``).

    Selected by ``--engine-mdo``: builds a ``MissionSpec``/``DesignVector`` from
    the shared CLI flags and runs the coupled solve — nozzle performance, regen
    cooling, pintle injector, and electric pump feed as ONE differentiable
    evaluation with the cooling Δp → pump-rise hydraulic edge closed.  Prints
    performance, the closed edge, the §3 mass ledger, and every constraint
    margin.  ``--engine-mdo-couple-cstar`` turns on the optional spray→η_c*
    feedback (default frozen — the RQ1 ablation reference).  jax is imported
    lazily so ordinary CLI runs never pay for it.
    """
    import dataclasses
    contour_request_error = _mdo_contour_request_error(args)
    if contour_request_error is not None:
        print(f"  {contour_request_error}")
        return 2

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from raosim.mdo.schema import DesignVector
    from raosim.mdo.engine import (
        ablation_delta,
        chamber_surfaces_for,
        solve_engine,
    )
    from raosim.mdo.constraints import (
        ENGINE_CONSTRAINT_SPECS,
        constraint_metadata,
        reason_text,
        status_from_rows,
    )
    from raosim.mdo.coolant_htd import (
        ModelCoverageError,
        require_htd_coverage,
    )
    from raosim.mdo.propellants import get_propellant
    from raosim.input_validation import WorkflowExitCode

    try:
        mission, _ = _mission_from_mdo_args(
            args, optimization_capable=False
        )
        surfaces = chamber_surfaces_for(mission)
        coolant_name = get_propellant(mission.propellant_name).coolant_name
        htd_available, htd_reason = require_htd_coverage(
            coolant_name,
            has_real_fluid_properties=False,
            allow_incomplete_physics=bool(args.allow_incomplete_physics),
        )
    except ModelCoverageError as exc:
        print(f"  model coverage incomplete: {exc}")
        return int(WorkflowExitCode.INDETERMINATE)
    except (KeyError, TypeError, ValueError) as exc:
        print(f"  MDO configuration rejected: {exc}")
        return int(WorkflowExitCode.INVALID_INPUT)
    chi_f = (args.fuel_injector_dp_fraction
             if args.fuel_injector_dp_fraction is not None else 0.20)
    chi_o = (args.oxidizer_injector_dp_fraction
             if args.oxidizer_injector_dp_fraction is not None else 0.20)
    film = args.film_frac if args.film_frac is not None else 0.0
    cw = args.channel_width if args.channel_width is not None else 5.0e-4
    ch = args.channel_height if args.channel_height is not None else 1.5e-3
    if args.film_slot_height is not None:
        mission = dataclasses.replace(
            mission, film_slot_height_default=float(args.film_slot_height))
    x = DesignVector(Pc=jnp.asarray(float(args.pc)),
                     eps=jnp.asarray(float(args.epsilon)),
                     dp_f_frac=jnp.asarray(chi_f), dp_o_frac=jnp.asarray(chi_o),
                     # These CLI controls are design variables in the MDO
                     # block, not merely architecture defaults on MissionSpec.
                     # Preserve thrust-scaled defaults when a flag is absent.
                     D_pintle=jnp.asarray(float(
                         args.pintle_diameter if args.pintle_diameter is not None
                         else mission.pintle_diameter)),
                     N_rpm=jnp.asarray(float(
                         args.pump_rpm if args.pump_rpm is not None
                         else mission.pump_speed_rpm)),
                     channel_width=jnp.asarray(float(cw)),
                     channel_height=jnp.asarray(float(ch)),
                     film_frac=jnp.asarray(float(film)),
                     t_wall=jnp.asarray(float(
                         args.t_wall if args.t_wall is not None else 8.0e-4)),
                     OF=jnp.asarray(float(mission.OF)),
                     layout=mission.design_layout())
    couple = bool(getattr(args, "engine_mdo_couple_cstar", False))
    r = solve_engine(
        x,
        mission,
        couple_eta_cstar=couple,
        surfaces=surfaces,
    )
    F = float

    print("=" * 66)
    print(" Whole-engine differentiable MDO  (raosim.mdo.engine, Phase 7)")
    print("=" * 66)
    print(f" propellant : {mission.propellant_name}   L*={mission.l_star:.3f} m   "
          f"coolant wall limit="
          + ("none (no coking)" if mission.rp1_coking_wall_temp_K > 5e3
             else f"{mission.rp1_coking_wall_temp_K:.0f} K"))
    print(f" design : Pc={F(x.Pc)/1e6:.2f} MPa  eps={F(x.eps):.1f}  "
          f"O/F={F(r.OF):.2f}  F={mission.thrust/1e3:.1f} kN  "
          f"chi_f={chi_f:.2f} chi_o={chi_o:.2f}")
    if getattr(args, "design_margins", False):
        print("          DESIGN MARGINS ON: heat flux x1.10, channel flow x0.90,"
              " film capacity x2 (SP-8087/Mirzamoghadam)")
    print(f"          film_frac={F(x.film_frac):.3f}  "
          f"channel={F(x.channel_width)*1e3:.2f}x{F(x.channel_height)*1e3:.2f} mm"
          f"  film slot={mission.film_slot_height_default*1e3:.2f} mm")
    print(f" eta_c* : {F(r.eta_cstar):.4f}  eta_CF={mission.eta_CF:.4f}  "
          f"({'coupled spray-TMR surrogate' if couple else 'frozen (default)'})")
    print(" -- performance ------------------------------------------------")
    print(f"    Rt = {F(r.Rt)*1e3:7.2f} mm     mdot = {F(r.mdot):6.3f} kg/s")
    print(f"    Cf = {F(r.Cf):7.3f} (ideal {F(r.Cf_ideal):.3f})  "
          f"Isp  = {F(r.Isp):6.1f} s")
    print(f"    Me = {F(r.Me):7.2f}        Pe   = {F(r.Pe)/1e3:6.1f} kPa   "
          f"(thrust resid {F(r.thrust_residual):+.1e})")
    print(" -- cooling  →  feed hydraulic edge (closed §5 loop) ------------")
    print(f"    jacket dp_regen = {F(r.dp_regen)/1e5:6.2f} bar  →  "
          f"fuel pump rise = {F(r.dp_rise_fuel)/1e5:6.2f} bar")
    print(f"    T_wg,max = {F(jnp.max(r.T_wg)):5.0f} K   "
          f"T_wc,max = {F(jnp.max(r.cooling.T_wc)):5.0f} K   "
          f"coolant out = {F(r.cooling.T_coolant_exit):5.0f} K")
    d = r.diagnostics
    print(f"    t_wall = {F(x.t_wall)*1e3:.2f} mm   sigma_thermal = "
          f"{F(d['sigma_thermal_max'])/1e6:.0f} MPa  (pressure bending "
          f"{F(d['sigma_pressure'])/1e6:.2f} MPa)")
    print(f"    coolant Mach = {F(d['coolant_mach']):.4f} (limit 0.35)   "
          f"v_cool = {F(d['coolant_velocity']):.2f} m/s   "
          f"eta_film = {F(d['eta_film_cooling']):.3f}")
    print(" -- injector ---------------------------------------------------")
    print(f"    TMR = {F(r.injector.momentum_ratio):.3f}   "
          f"spray half-angle = {F(r.injector.spray_half_angle_deg):.1f} deg   "
          f"BF = {F(r.injector.blockage_factor):.3f}")
    print(" -- electric feed ----------------------------------------------")
    print(f"    P_electric = {F(r.feed.P_electric_total)/1e3:.2f} kW    "
          f"battery E/P = {F(r.feed.battery.energy_limited_mass):.1f}"
          f"/{F(r.feed.battery.power_limited_mass):.1f} kg")
    print(" -- mass ledger [kg] -------------------------------------------")
    for k, v in r.mass_ledger.items():
        if k.endswith("_placeholder"):
            print(f"    {k.removesuffix('_placeholder'):30s} {'unavailable':>12s}")
        else:
            print(f"    {k:30s} {F(v):8.2f}")
    print(
        f"    {'ELECTRIC PACKAGE EXACT':30s} "
        f"{F(r.electric_package_exact_mass):8.2f}"
    )
    print(
        f"    {'ELECTRIC PACKAGE OBJECTIVE':30s} "
        f"{F(r.electric_package_objective_mass):8.2f}"
    )
    print(
        f"    {'PARTIAL DRY MASS EXACT':30s} "
        f"{F(r.dry_mass_partial_exact_mass):8.2f}"
    )
    print(
        f"    {'PARTIAL DRY MASS OBJECTIVE':30s} "
        f"{F(r.dry_mass_partial_objective_mass):8.2f}"
    )
    print(" -- constraint margins (>= 0 feasible) -------------------------")
    applicable, available, required, reasons = constraint_metadata(
        mission, surfaces
    )
    constraint_values = [
        F(r.constraints[spec.engine_key])
        for spec in ENGINE_CONSTRAINT_SPECS
    ]
    physics_status = status_from_rows(
        constraint_values, applicable, available, required,
        nonfinite="unknown",
    )
    for i, (spec, value) in enumerate(
        zip(ENGINE_CONSTRAINT_SPECS, constraint_values)
    ):
        if not applicable[i]:
            suffix = reason_text(reasons[i], spec, mission)
            print(f"    {spec.engine_key:30s} {'not applicable':>12s}   {suffix}")
        elif not available[i]:
            suffix = reason_text(reasons[i], spec, mission)
            print(f"    {spec.engine_key:30s} {'unavailable':>12s}   {suffix}")
        else:
            print(
                f"    {spec.engine_key:30s} {value:+.4g}"
                f"{'   <-- VIOLATED' if value < 0.0 else ''}"
            )
    print(f"    physics verdict: {physics_status.upper()}")
    if not htd_available and htd_reason:
        print(f"    screening limitation: {htd_reason}")
    if getattr(args, "mdo_export", False):
        from raosim.mdo.postprocess import reevaluate, summarise
        print(" -- authoritative re-evaluation (Phase 11) --------------------")
        dd = {k: float(v) for k, v in x.as_dict().items()}
        # Native IPT is not an authoritative v2 output; STEP is the source
        # geometry for both ``ipt`` and ``both`` requests.
        authoritative_cad = (
            "step" if args.cad in {"step", "ipt", "both"} else "none"
        )
        (
            authoritative_contour,
            host_rao_solver_options,
        ) = _mdo_authoritative_contour_handoff(args)
        rv = reevaluate(
            dd,
            mission,
            mdo_result=r,
            mdo_summary={
                "Isp": F(r.Isp),
                "Rt": F(r.Rt),
                "eps": F(x.eps),
                "mdot": F(r.mdot),
                "thrust": F(mission.thrust),
            },
            optimizer_metadata={
                "workflow": "single_design_evaluation",
                "couple_eta_cstar": couple,
                "authoritative_contour_requested": authoritative_contour,
            },
            output_dir=args.out,
            cad=authoritative_cad,
            contour_method=authoritative_contour,
            host_rao_solver_options=host_rao_solver_options,
        )
        print(summarise(rv))
        c = rv.result.contour
        print(
            f"    authoritative host contour ({c.get('method', 'unknown')}): "
            f"{len(c['x'])} points, "
              f"Rt={c['Rt']*1e3:.2f} mm, exit r={max(c['y'])*1e3:.2f} mm")
        print("    authoritative snapshot/report: "
              f"{rv.metadata['authoritative_snapshot_report']}")
    if not couple:
        d = F(ablation_delta(x, mission, "Isp"))
        print(" -- RQ1 ablation -----------------------------------------------")
        print(f"    dIsp(couple eta_c*) = {d:+.2f} s  "
              "(bound on the spray→c* correlation)")
    print("=" * 66)
    if not bool(r.solver_converged) or not bool(r.finite):
        return int(WorkflowExitCode.SOLVER_FAILED)
    if physics_status == "fail":
        return int(WorkflowExitCode.CANDIDATE_VIOLATES)
    if physics_status == "unknown":
        return int(WorkflowExitCode.INDETERMINATE)
    return int(WorkflowExitCode.MET)


def _parse_thrust_condition(raw: str):
    """CLI spelling of a thrust condition -> the requirements-layer form."""

    text = str(raw).strip().lower()
    if text.startswith("altitude"):
        _, _, value = text.partition(":")
        if not value:
            raise ValueError(
                "--thrust-condition altitude needs a height, e.g. "
                "'altitude:12000'"
            )
        return ("altitude", float(value))
    return text


def _run_requirements(args) -> int:
    """Requirement-driven design (``raosim.requirements``, Layer 0).

    The user states SP-125 §2.1 targets; the optimiser chooses the design
    variables.  The coverage table is printed *before* the numbers, because a
    performance figure produced against a partially screened requirement means
    something weaker than it looks and the reader needs to know that first.
    """

    from raosim.requirements import (
        EngineRequirement,
        RequirementAnalysisConfig,
        solve_requirement,
    )
    from raosim.mdo.coolant_htd import ModelCoverageError

    throttle = None
    if args.throttle_range:
        lo, _, hi = str(args.throttle_range).partition(",")
        throttle = (float(lo), float(hi))

    try:
        of_mode, mixture_ratio = _resolve_mdo_of_intent(
            args, optimization_capable=True
        )
        analysis_config = RequirementAnalysisConfig(
            chamber_property_table=args.mdo_chamber_property_table,
            length_pct=float(args.length_pct),
            liner_material=args.material,
        )
        req = EngineRequirement(
            thrust=float(args.target_thrust if args.target_thrust is not None
                         else 13.0e3),
            thrust_condition=_parse_thrust_condition(args.thrust_condition),
            isp_min=args.isp_min,
            isp_basis=args.isp_basis,
            flight_duration=float(
                args.flight_duration if args.flight_duration is not None
                else args.burn_time),
            qualification_duration=args.qualification_duration,
            mixture_ratio_mode=of_mode,
            mixture_ratio=mixture_ratio,
            burnout_mass_max=args.burnout_mass_max,
            envelope_diameter_max=args.envelope_diameter_max,
            envelope_length_max=args.envelope_length_max,
            propellant=(getattr(args, "mdo_propellant", None) or "LOX/RP-1"),
            throttle_range=throttle,
            reusable_cycles=args.reusable_cycles,
            objective=args.mdo_mass_objective,
            analysis_config=analysis_config,
        )
    except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
        print(f"  requirement rejected: {exc}")
        return 2

    print("=" * 78)
    print(" Requirement-driven design  (raosim.requirements, SP-125 §2.1)")
    print("=" * 78)

    try:
        result = solve_requirement(
            req,
            couple_eta_cstar=bool(args.engine_mdo_couple_cstar),
            allow_incomplete_physics=bool(args.allow_incomplete_physics),
        )
    except ModelCoverageError as exc:
        print(f"  model coverage incomplete: {exc}")
        return 3
    except (KeyError, TypeError, ValueError, NotImplementedError) as exc:
        print(f"  requirement rejected: {exc}")
        return 2
    except Exception as exc:
        print(f"  requirement solver failed: {type(exc).__name__}: {exc}")
        return 4
    print(result.summary())

    n = result.nlp
    d = n.constraints
    print()
    print(" design chosen by the optimiser:")
    for key, fmt, scale, unit in (
        ("Pc", "%8.3f", 1e-6, "MPa"), ("eps", "%8.3f", 1.0, "-"),
        ("film_frac", "%8.3f", 1.0, "-"), ("t_wall", "%8.3f", 1e3, "mm"),
        ("channel_width", "%8.3f", 1e3, "mm"),
        ("channel_height", "%8.3f", 1e3, "mm"),
        ("D_pintle", "%8.3f", 1e3, "mm"), ("N_rpm", "%8.1f", 1e-3, "krpm"),
    ):
        print(f"   {key:<16}" + fmt % (n.design[key] * scale) + f" {unit}")

    # Fractional margins read as "fraction of the allowance left"; print the
    # physical side too, so the requirement and the hardware are both legible.
    print()
    print(" requirement utilisation (screened quantities; lower bounds):")
    for name, limit in (("envelope_diameter", req.envelope_diameter_max),
                        ("envelope_length", req.envelope_length_max),
                        ("dry_mass_partial", req.burnout_mass_max)):
        if limit is None:
            continue
        used = (1.0 - d[name]) * limit
        print(f"   {name:<20} {used:9.4f} of {limit:9.4f}"
              f"   ({100.0 * (1.0 - d[name]):5.1f} % used)")
    print("=" * 78)
    if not n.success:
        return 4
    if result.requirements_met is True:
        return 0
    if result.requirements_met is False:
        return 1
    return 3


def _run_engine_mdo_optimize(args) -> int:
    """ε-constraint hard-constrained whole-engine MDO (``raosim.mdo.nlp``).

    Selected by ``--engine-mdo-optimize``: the user supplies the mission
    requirements (thrust, O/F, burn time, ambient) and an Isp target, and the
    optimiser solves for the DESIGN (Pc, eps, injector Δp fractions, D_pintle,
    pump rpm) that minimises an explicitly named resolved mass subtotal s.t.
    Isp ≥ floor and every enforced discipline margin ≥ 0, with exact JAX
    Jacobians (SLSQP). Exact electric and partial-dry subtotals are reported.
    ``--isp-sweep LO,HI,N`` traces the mass–Isp Pareto frontier; ``--isp-min``
    does a single min-mass solve.  jax is imported lazily.
    """
    contour_request_error = _mdo_contour_request_error(args)
    if contour_request_error is not None:
        print(f"  {contour_request_error}")
        return 2

    import jax
    jax.config.update("jax_enable_x64", True)
    from raosim.mdo.nlp import solve_min_mass, pareto_frontier, DEFAULT_ENFORCED
    from raosim.mdo.coolant_htd import ModelCoverageError
    try:
        mission, of_mode = _mission_from_mdo_args(
            args, optimization_capable=True
        )
    except (KeyError, TypeError, ValueError) as exc:
        print(f"  MDO configuration rejected: {exc}")
        return 2
    objective = args.mdo_mass_objective
    allow_incomplete = bool(args.allow_incomplete_physics)
    couple = bool(args.engine_mdo_couple_cstar)
    (
        authoritative_contour,
        host_rao_solver_options,
    ) = _mdo_authoritative_contour_handoff(args)

    print("=" * 78)
    print(" Whole-engine ε-constraint MDO  (raosim.mdo.nlp, Phase 8/9)")
    print("=" * 78)
    print(f" propellant : {mission.propellant_name}  (L*={mission.l_star:.3f} m)")
    print(f" mission : F={mission.thrust/1e3:.1f} kN  O/F seed={mission.OF:.2f}  "
          f"Pa={mission.Pa/1e3:.1f} kPa  burn={mission.burn_time:.0f} s   "
          f"O/F mode={of_mode.value}  "
          f"eta_c*={'coupled' if couple else 'frozen'}")
    print(f" objective: {objective}; exact resolved subtotals also reported  "
          "enforced margins ≥ 0: "
          f"{', '.join(DEFAULT_ENFORCED)}")
    if args.design_margins:
        print(" margins: SP-8087/Mirzamoghadam DESIGN MARGINS ON (heat flux "
              "x1.10, flow x0.90, film capacity x2)")
    print(" note: coking is ENFORCED via film cooling (design var film_frac) —"
          " the wall is\n       coolant-enthalpy-limited, so film (not channel"
          " geometry) is the coking lever;\n       film costs c*/Isp, so the"
          " frontier is genuinely thermal-limited.")
    print("-" * 78)

    def _fmt(r):
        d = r.design
        exact_selected = (
            r.exact_dry_mass_partial
            if r.objective_name == "min_dry_mass_partial"
            else r.exact_electric_package_mass
        )
        return (f" Isp>={r.isp_min:6.1f} | {r.objective_name}="
                f"{r.objective_mass:7.2f} kg exact={exact_selected:7.2f} kg  "
                f"Isp={r.Isp:6.1f} s | Pc={d['Pc']/1e6:4.2f}MPa eps={d['eps']:5.2f} "
                f"film={d['film_frac']:.3f} t_w={d['t_wall']*1e3:.2f}mm "
                f"N={d['N_rpm']/1e3:4.1f}k | physics={r.physics_status:7} "
                f"cok={r.constraints['coking']:+.0f}")

    if args.isp_sweep:
        try:
            lo, hi, n = args.isp_sweep.split(",")
            lo, hi, n = float(lo), float(hi), int(n)
            grid = [lo + (hi - lo) * i / max(n - 1, 1) for i in range(n)]
        except Exception:
            print("  --isp-sweep must be LO,HI,N  (e.g. 250,320,6)")
            return 2
        print(f" Pareto frontier over Isp floors {grid[0]:.0f}..{grid[-1]:.0f} "
              f"({len(grid)} pts; warm-started):")
        try:
            frontier = pareto_frontier(
                mission,
                grid,
                couple_eta_cstar=couple,
                allow_incomplete_physics=allow_incomplete,
                objective=objective,
            )
        except ModelCoverageError as exc:
            print(f"  model coverage incomplete: {exc}")
            return 3
        except (KeyError, TypeError, ValueError) as exc:
            print(f"  MDO configuration rejected: {exc}")
            return 2
        except Exception as exc:
            print(f"  MDO solver failed: {type(exc).__name__}: {exc}")
            return 4
        for index, r in enumerate(frontier):
            print(_fmt(r))
            if args.mdo_export and r.feasible:
                from raosim.mdo.postprocess import reevaluate, summarise

                authoritative_cad = (
                    "step"
                    if args.cad in {"step", "ipt", "both"}
                    else "none"
                )
                point_out = args.out / f"pareto_{index:03d}"
                rv = reevaluate(
                    r.design,
                    mission,
                    mdo_summary={
                        "Isp": r.Isp,
                        "eps": r.design["eps"],
                        "thrust": mission.thrust,
                    },
                    optimizer_metadata={
                        "workflow": "epsilon_constraint_pareto_point",
                        "pareto_index": index,
                        "pareto_count": len(frontier),
                        "method": "SLSQP",
                        "success": r.success,
                        "feasible": r.feasible,
                        "iterations": r.n_iter,
                        "message": r.message,
                        "isp_min_s": r.isp_min,
                        "max_violation": r.max_violation,
                        "constraints": dict(r.constraints),
                        "enforced": list(r.enforced),
                        "design": dict(r.design),
                        "objective_name": r.objective_name,
                        "objective_mass_kg": r.objective_mass,
                        "electric_package_objective_mass_kg": (
                            r.electric_package_objective_mass
                        ),
                        "dry_mass_partial_objective_mass_kg": (
                            r.dry_mass_partial_objective_mass
                        ),
                        "exact_electric_package_mass_kg": (
                            r.exact_electric_package_mass
                        ),
                        "exact_dry_mass_partial_kg": r.exact_dry_mass_partial,
                        "specific_impulse_s": r.Isp,
                        "couple_eta_cstar": couple,
                        "authoritative_contour_requested": (
                            authoritative_contour
                        ),
                    },
                    couple_eta_cstar=couple,
                    output_dir=point_out,
                    cad=authoritative_cad,
                    contour_method=authoritative_contour,
                    host_rao_solver_options=host_rao_solver_options,
                )
                print(summarise(rv))
                print(
                    "    authoritative snapshot/report: "
                    f"{rv.metadata['authoritative_snapshot_report']}"
                )
            elif args.mdo_export:
                print(
                    "    authoritative re-evaluation skipped: Pareto point "
                    "is not MDO-feasible"
                )
    else:
        isp_min = args.isp_min if args.isp_min is not None else 230.0
        try:
            r = solve_min_mass(
                mission,
                float(isp_min),
                couple_eta_cstar=couple,
                allow_incomplete_physics=allow_incomplete,
                objective=objective,
            )
        except ModelCoverageError as exc:
            print(f"  model coverage incomplete: {exc}")
            return 3
        except (KeyError, TypeError, ValueError) as exc:
            print(f"  MDO configuration rejected: {exc}")
            return 2
        except Exception as exc:
            print(f"  MDO solver failed: {type(exc).__name__}: {exc}")
            return 4
        print(_fmt(r))
        if args.mdo_export:
            from raosim.mdo.postprocess import reevaluate, summarise
            print(" -- authoritative re-evaluation (Phase 11) ----------------")
            authoritative_cad = (
                "step" if args.cad in {"step", "ipt", "both"} else "none"
            )
            rv = reevaluate(
                r.design,
                mission,
                mdo_summary={
                    "Isp": r.Isp,
                    "eps": r.design["eps"],
                    "thrust": mission.thrust,
                },
                optimizer_metadata={
                    "workflow": "epsilon_constraint_optimization",
                    "method": "SLSQP",
                    "success": r.success,
                    "feasible": r.feasible,
                    "iterations": r.n_iter,
                    "message": r.message,
                    "isp_min_s": r.isp_min,
                    "max_violation": r.max_violation,
                    "constraints": dict(r.constraints),
                    "enforced": list(r.enforced),
                    "design": dict(r.design),
                    "objective_name": r.objective_name,
                    "objective_mass_kg": r.objective_mass,
                    "electric_package_objective_mass_kg": (
                        r.electric_package_objective_mass
                    ),
                    "dry_mass_partial_objective_mass_kg": (
                        r.dry_mass_partial_objective_mass
                    ),
                    "exact_electric_package_mass_kg": (
                        r.exact_electric_package_mass
                    ),
                    "exact_dry_mass_partial_kg": r.exact_dry_mass_partial,
                    "specific_impulse_s": r.Isp,
                    "couple_eta_cstar": couple,
                    "authoritative_contour_requested": (
                        authoritative_contour
                    ),
                },
                couple_eta_cstar=couple,
                output_dir=args.out,
                cad=authoritative_cad,
                contour_method=authoritative_contour,
                host_rao_solver_options=host_rao_solver_options,
            )
            print(summarise(rv))
            print("    authoritative snapshot/report: "
                  f"{rv.metadata['authoritative_snapshot_report']}")
        print(f"  solver: {r.message}  "
              f"(iters={r.n_iter}, max_violation={r.max_violation:.1e})")
    print("=" * 78)
    results = frontier if args.isp_sweep else [r]
    if any(not result.success for result in results):
        return 4
    if any(result.physics_status == "unknown" for result in results):
        return 3
    if any(not result.optimizer_feasible for result in results):
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    from raosim.pumps import SCREENING_DEFAULTS as PUMP_DEFAULTS

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    # nozzle
    ap.add_argument("--rt", type=float, default=0.020, help="throat radius [m]")
    ap.add_argument("--target-thrust", type=float, default=None,
                    help="design thrust [N]; sizes Rt from F/(Cf·Pc). "
                         "Mutually exclusive with an explicit --rt.")
    ap.add_argument("--epsilon", type=float, default=10.0)
    ap.add_argument("--length-pct", type=float, default=80.0)
    ap.add_argument(
        "--contour-method", choices=("bezier", "rao-bvp"), default="bezier",
        help="bezier = trusted deterministic Rao/TOP chart contour (default); "
             "rao-bvp = experimental variational/MOC solve",
    )
    ap.add_argument(
        "--allow-chart-extrapolation", action="store_true",
        help="write diagnostic Bezier artifacts when epsilon/length lies "
             "outside the digitized Rao/TOP chart domain; such artifacts are "
             "not benchmark-qualified",
    )
    ap.add_argument("--gamma", type=float, default=1.24,
                    help="combustion-product gamma; overridden by the resolved "
                         "propellant unless passed explicitly")
    ap.add_argument("--pa-over-p0", type=float, default=0.01,
                    help="design ambient/chamber pressure ratio (sets Pa)")
    # propellant + thermochemistry (drives gamma, c*, Tc, mass flow)
    ap.add_argument("--thermo-mode", choices=("cea", "constant-gamma"),
                    default="constant-gamma",
                    help="cea = external RocketCEA chamber-state snapshot "
                         "when installed (nozzle expansion remains frozen "
                         "constant-gamma; not local physical validation); "
                         "constant-gamma = built-in literature table "
                         "(screening)")
    ap.add_argument(
        "--nozzle-expansion-model",
        choices=("constant-gamma", "frozen-variable-cp"),
        default="constant-gamma",
        help=(
            "constant-gamma = existing calorically-perfect expansion; "
            "frozen-variable-cp = thermally-perfect, fixed-composition "
            "quasi-1-D expansion from --frozen-gas-table (Bezier only)"
        ),
    )
    ap.add_argument(
        "--frozen-gas-table",
        type=Path,
        default=None,
        help=(
            "strict, provenance-bound JSON cp(T) table required by "
            "--nozzle-expansion-model frozen-variable-cp"
        ),
    )
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
    ap.add_argument(
        "--spray-cstar-coupling", action="store_true",
        help="opt into a relaxed fixed point from injector vaporization to "
             "eta_cstar and cycle mass flow; requires --injector pintle plus "
             "explicit --spray-eta-mixing/--spray-eta-combustion",
    )
    ap.add_argument("--spray-eta-mixing", type=float, default=None,
                    help="independently supplied spray mixing efficiency (0,1]")
    ap.add_argument("--spray-eta-combustion", type=float, default=None,
                    help="independently supplied chemical-completion efficiency (0,1]")
    ap.add_argument("--spray-coupling-relaxation", type=float, default=0.5,
                    help="relaxation factor for the spray/c-star fixed point")
    ap.add_argument("--spray-coupling-tolerance", type=float, default=1.0e-4,
                    help="relative eta_cstar closure tolerance")
    ap.add_argument("--spray-coupling-max-iterations", type=int, default=25,
                    help="maximum spray/c-star fixed-point iterations")
    ap.add_argument(
        "--spray-evaporation-constant", type=float, default=1.0e-6,
        help="calibrated d-squared-law evaporation constant [m^2/s]; default "
             "1e-6 is only a hydrocarbon-class screening value",
    )
    # chamber and shared throat
    ap.add_argument("--l-star", type=float, default=1.0,
                    help="chamber characteristic length Vc/At [m]")
    ap.add_argument("--contraction-ratio", type=float, default=2.5,
                    help="chamber/throat area ratio Ac/At")
    ap.add_argument("--shoulder-radius-factor", type=float, default=None,
                    help="chamber shoulder radius / Rt; used only with "
                         "--shoulder-sizing scalar. When omitted in scalar "
                         "mode, the legacy placeholder is 0.25.")
    ap.add_argument("--shoulder-sizing", choices=("scalar", "auto"),
                    default="auto",
                    help="auto: derive the fillet geometrically as the "
                         "smoothest contraction the contour allows for the "
                         "given Rt, contraction ratio, convergent angle and Ru "
                         "(default; see docs/shoulder_radius_design_basis.md). "
                         "scalar: use --shoulder-radius-factor or the legacy "
                         "0.25 placeholder.")
    ap.add_argument("--shoulder-fill-fraction", type=float, default=0.8,
                    help="with --shoulder-sizing auto, fraction of the maximum "
                         "feasible fillet to use (0<f<1; 0.8 keeps a ~20%% "
                         "straight convergent cone)")
    ap.add_argument("--minimum-cylindrical-length", type=float, default=None,
                    help="minimum cylindrical chamber length [m]; geometric "
                         "floor 1e-6 when omitted")
    ap.add_argument("--convergent-angle", type=float, default=45.0,
                    help="shared chamber/nozzle convergent half-angle [deg]")
    ap.add_argument("--ru-factor", type=float, default=1.5,
                    help="shared upstream throat radius / Rt")
    ap.add_argument("--cd-target", type=float, default=None,
                    help="derive --ru-factor from a target inviscid throat "
                         "discharge coefficient using Hall's leading-order "
                         "transonic relation, bounded to the cited SP-8120 "
                         "Ru/Rt range 0.6-1.5")
    ap.add_argument(
        "--allow-throat-radius-extension", action="store_true",
        help="allow the repository-only diagnostic Ru/Rt extension from the "
             "SP-8120 upper bound 1.5 to 2.0",
    )
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
    ap.add_argument(
        "--release-evidence-manifest", type=Path, default=None,
        help="versioned JSON manifest containing traceable CFD/FEA/drawing/"
             "proof/cold-flow/hot-fire evidence for the exported configuration",
    )
    ap.add_argument(
        "--configuration-id", default=None,
        help="configuration-controlled design/hardware identifier that must "
             "match every release-evidence record",
    )
    ap.add_argument(
        "--require-release-evidence", action="store_true",
        help="block the run before artifact generation unless every physical-"
             "release evidence requirement in the manifest passes",
    )
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
                         "default Pc plus the fuel injector dP when the "
                         "coolant is the cycle fuel")
    ap.add_argument("--injector-pressure-drop", type=float, default=None,
                    help=argparse.SUPPRESS)
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
    ap.add_argument(
                    "--injector-architecture",
                    choices=("fixed_discrete", "son_continuous_movable"),
                    default="fixed_discrete",
                    help="fixed slots/holes, or the Son 2017 continuous "
                         "movable centre-pintle architecture")
    ap.add_argument("--injector-sizing", choices=("auto", "fixed", "movable"),
                    default="auto",
                    help="auto: derive openings from ṁ/ΔP; fixed: evaluate "
                         "supplied geometry without resizing; movable: hold "
                         "the axial annulus fixed and solve the Son opening")
    ap.add_argument("--pintle-radial-stream", choices=("fuel", "oxidizer"),
                    default="fuel", help="which stream uses the radial holes "
                                         "or slots")
    ap.add_argument("--fuel-discharge-coefficient", type=float, default=0.7,
                    help="fuel metering discharge coefficient Cd")
    ap.add_argument("--oxidizer-discharge-coefficient", type=float, default=0.7,
                    help="oxidizer metering discharge coefficient Cd")
    ap.add_argument("--pintle-diameter", type=float, default=None,
                    help="pintle diameter [m] (annulus + slot anchor); "
                         "default 0.30·chamber diameter")
    ap.add_argument("--pintle-slot-count", type=int, default=24,
                    help="number of radial slots/holes")
    ap.add_argument("--pintle-radial-exit",
                    choices=("holes", "slots", "continuous_radial_gap"),
                    default="holes",
                    help="coaxial pintle tip radial exit style: drilled round "
                         "jets (holes, default), rectangular slots, or the "
                         "Son continuous movable gap")
    ap.add_argument("--pintle-slot-aspect-ratio", type=float, default=1.0,
                    help="slot height/width for auto-sizing")
    ap.add_argument("--pintle-deflector-angle", type=float, default=0.0,
                    help="radial-stream deflector angle [deg]")
    ap.add_argument("--pintle-target-momentum-ratio", type=float, default=None,
                    help="optional target radial/axial momentum ratio; in "
                         "auto sizing, solves the radial stream dP/Pc over "
                         "the configured injector design envelope")
    ap.add_argument("--pintle-impingement-distance", type=float, default=None,
                    help="distance from openings to stream interaction [m]")
    ap.add_argument("--injector-min-feature", type=float, default=3.0e-4,
                    help="manufacturing floor for gaps/slots/ligaments [m]")
    ap.add_argument("--allow-infeasible-injector", action="store_true",
                    help="export the chamber even when injector gates fail "
                         "(default: failing gates block export, exit nonzero)")
    # feed-system ledger inputs (also consumed by --electric-pump)
    ap.add_argument("--feed-architecture", choices=("pump_fed", "pressure_fed"),
                    default="pump_fed",
                    help="feed architecture label stored in the feed ledger")
    ap.add_argument("--fuel-supply-pressure", type=float, default=None,
                    help="available fuel pump/tank outlet pressure [Pa]")
    ap.add_argument("--oxidizer-supply-pressure", type=float, default=None,
                    help="available oxidizer pump/tank outlet pressure [Pa]")
    ap.add_argument("--fuel-flow-capacity", type=float, default=None,
                    help="available fuel pump/feed capacity [kg/s]")
    ap.add_argument("--oxidizer-flow-capacity", type=float, default=None,
                    help="available oxidizer pump/feed capacity [kg/s]")
    ap.add_argument("--fuel-line-loss", type=float, default=0.0,
                    help="fuel line/valve/filter loss charged to pump [Pa]")
    ap.add_argument("--oxidizer-line-loss", type=float, default=0.0,
                    help="oxidizer line/valve/filter loss charged to pump [Pa]")
    ap.add_argument("--fuel-line-loss-fraction", type=float, default=0.0,
                    help="fuel line loss as a fraction of Pc")
    ap.add_argument("--oxidizer-line-loss-fraction", type=float, default=0.0,
                    help="oxidizer line loss as a fraction of Pc")
    ap.add_argument("--fuel-manifold-loss", type=float, default=0.0,
                    help="fuel manifold loss allowance charged to pump [Pa]")
    ap.add_argument("--oxidizer-manifold-loss", type=float, default=0.0,
                    help="oxidizer manifold loss allowance charged to pump [Pa]")
    ap.add_argument("--fuel-manifold-loss-fraction", type=float, default=0.0,
                    help="fuel manifold allowance as a fraction of Pc")
    ap.add_argument("--oxidizer-manifold-loss-fraction", type=float, default=0.0,
                    help="oxidizer manifold allowance as a fraction of Pc")
    ap.add_argument("--fuel-control-margin", type=float, default=0.0,
                    help="fuel feed control/transient margin [Pa]")
    ap.add_argument("--oxidizer-control-margin", type=float, default=0.0,
                    help="oxidizer feed control/transient margin [Pa]")
    ap.add_argument("--fuel-control-margin-fraction", type=float, default=0.0,
                    help="fuel control margin as a fraction of Pc")
    ap.add_argument("--oxidizer-control-margin-fraction", type=float, default=0.0,
                    help="oxidizer control margin as a fraction of Pc")
    ap.add_argument("--fuel-tank-pressure", type=float, default=None,
                    help="fuel pump inlet/tank pressure for head/NPSH [Pa]")
    ap.add_argument("--oxidizer-tank-pressure", type=float, default=None,
                    help="oxidizer pump inlet/tank pressure for head/NPSH [Pa]")
    ap.add_argument("--fuel-npsh-required", type=float, default=None,
                    help="fuel pump required NPSH pressure margin [Pa]")
    ap.add_argument("--oxidizer-npsh-required", type=float, default=None,
                    help="oxidizer pump required NPSH pressure margin [Pa]")
    ap.add_argument("--pump-efficiency-fuel", type=float, default=None,
                    help="fuel pump efficiency override; default auto-estimates "
                         "from pump flow/head duty")
    ap.add_argument("--pump-efficiency-oxidizer", type=float, default=None,
                    help="oxidizer pump efficiency override; default auto-estimates "
                         "from pump flow/head duty")
    # electric pump sizing
    ap.add_argument("--electric-pump", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="size electric pump drive, battery, impeller, inducer, "
                         "and diffuser; use --no-electric-pump to disable the "
                         "complete-package default")
    ap.add_argument("--pump-rpm", type=float, default=None,
                    help="pump shaft speed override [rpm]; default solves from "
                         "specific speed and impeller geometry bounds")
    ap.add_argument("--pump-max-rpm", type=float, default=120000.0,
                    help="selected motor/pump maximum speed [rpm]")
    ap.add_argument("--burn-time", type=float, default=10.0,
                    help="burn duration for battery energy sizing [s]")
    ap.add_argument("--motor-voltage", type=float, default=None,
                    help="motor/inverter DC bus voltage [V]; default selects a "
                         "standard bus from power/current requirements")
    ap.add_argument("--motor-efficiency", type=float,
                    default=PUMP_DEFAULTS["motor_efficiency"],
                    help="motor efficiency for electric pump sizing")
    ap.add_argument("--inverter-efficiency", type=float,
                    default=PUMP_DEFAULTS["inverter_efficiency"],
                    help="inverter efficiency for electric pump sizing")
    ap.add_argument("--motor-power-density", type=float,
                    default=PUMP_DEFAULTS["motor_power_density"],
                    help="motor power density [W/kg]")
    ap.add_argument("--inverter-power-density", type=float,
                    default=PUMP_DEFAULTS["inverter_power_density"],
                    help="inverter/controller power density [W/kg]")
    ap.add_argument("--motor-max-power", type=float, default=None,
                    help="per-stream motor shaft-power limit [W]")
    ap.add_argument("--motor-max-current", type=float, default=None,
                    help="per-stream drive current limit [A]")
    ap.add_argument("--motor-torque-limit", type=float, default=None,
                    help="per-stream shaft torque limit [N m]")
    ap.add_argument("--motor-heat-rejection-limit", type=float, default=None,
                    help="per-stream motor+inverter heat rejection limit [W]")
    ap.add_argument("--battery-energy-density", type=float,
                    default=PUMP_DEFAULTS["battery_energy_density"],
                    help="pack-level usable energy density [J/kg]")
    ap.add_argument("--battery-power-density", type=float,
                    default=PUMP_DEFAULTS["battery_power_density"],
                    help="pack-level discharge power density [W/kg]")
    ap.add_argument("--battery-discharge-efficiency", type=float,
                    default=PUMP_DEFAULTS["battery_discharge_efficiency"],
                    help="battery discharge efficiency")
    ap.add_argument("--battery-structural-margin", type=float,
                    default=PUMP_DEFAULTS["battery_structural_margin"],
                    help="battery mass multiplier for packaging/structure")
    ap.add_argument("--battery-voltage", type=float, default=None,
                    help="battery pack voltage [V]; defaults to --motor-voltage")
    ap.add_argument("--battery-max-current", type=float, default=None,
                    help="battery/controller current limit [A]")
    ap.add_argument("--vehicle-mass", type=float, default=None,
                    help="vehicle gross/liftoff mass for battery mass fraction [kg]")
    ap.add_argument("--battery-max-mass-fraction", type=float, default=None,
                    help="maximum acceptable battery/vehicle mass fraction")
    ap.add_argument("--pump-head-coefficient", type=float, default=0.55,
                    help="centrifugal impeller head coefficient psi")
    ap.add_argument("--pump-flow-coefficient", type=float, default=0.08,
                    help="centrifugal impeller flow coefficient phi")
    ap.add_argument("--pump-tip-speed-limit", type=float, default=350.0,
                    help="screening impeller material/fabrication tip-speed limit [m/s]")
    ap.add_argument("--pump-max-head-per-stage", type=float, default=2500.0,
                    help="screening maximum head per centrifugal stage [m]")
    ap.add_argument("--pump-visualize", action="store_true",
                    help="save pump_particles.gif for the sized electric pump")
    ap.add_argument("--pump-cad", choices=("none", "auto", "reference", "parts"),
                    default="auto",
                    help="electric-pump CAD/reference package: none skips CAD; "
                         "auto/parts writes per-component CAD for impeller, "
                         "inducer, diffuser/volute, motor, inverter, and "
                         "battery when geometry is solved; reference also "
                         "writes the assembly reference")
    ap.add_argument("--pump-cad-format", choices=("stl", "step", "both"),
                    default="stl",
                    help="pump CAD exchange format; stl = faceted mesh "
                         "package, step = true CadQuery B-rep parts + named "
                         "assemblies (requires cadquery; the old faceted "
                         "pseudo-STEP was removed), both = stl and step")
    ap.add_argument("--allow-open-pump-mesh", action="store_true",
                    help="export pump part STLs even when the mesh gate finds "
                         "boundary/non-manifold edges or bad winding "
                         "(default: fail like the wall STL gate)")
    ap.add_argument("--engine-assembly", action="store_true",
                    help="assemble the exported wall/jacket/pintle/pump STEP "
                         "artifacts into one engine_assembly.step (requires "
                         "CadQuery; layout placement, not routed feed lines)")
    ap.add_argument("--allow-infeasible-pump", action="store_true",
                    help="continue exporting when electric-pump gates fail")
    ap.add_argument("--throttle-map", default=None,
                    help="comma-separated throttle levels in (0,1]. Fixed "
                         "discrete injectors receive a non-kinematic area "
                         "study; Son movable injectors hold hardware fixed "
                         "and solve physical travel plus the required separate "
                         "upstream annulus controller")
    ap.add_argument("--throttle-pc-exponent", type=float, default=1.0,
                    help="Pc(f)=Pc·f^exp for the throttle map (1=Pc∝ṁ, "
                         "0=constant Pc)")
    ap.add_argument("--injector-cad",
                    choices=("none", "auto", "reference", "parts", "step",
                             "machined"),
                    default="auto",
                    help="pintle package CAD mode with --injector pintle: "
                         "none writes only JSON/CSV/SVG/PNG; reference writes a "
                         "single CAD-neutral reference file; parts also writes "
                         "named part files; auto writes the machined STEP package "
                         "when CadQuery is available and otherwise keeps the "
                         "mandatory package plus manufacturing report; "
                         "machined writes Boolean-cut STEP bodies and a "
                         "manufacturing report; step is the legacy alias for "
                         "machined STEP output")
    ap.add_argument("--injector-cad-format", choices=("step", "stl", "dxf"),
                    default="step",
                    help="format for --injector-cad reference/parts/auto "
                         "(STEP default; DXF is a 2-D meridional profile)")
    ap.add_argument("--pintle-sleeve", action="store_true",
                    help="include the movable sleeve body in the pintle CAD")
    # Son et al. (2017) continuous movable-pintle geometry, calibration,
    # metrology, and static actuator evidence.  These stay separate from the
    # fixed slot/hole CAD fields above.
    ap.add_argument("--movable-post-diameter", type=float, default=None,
                    help="Son movable post outside diameter D_post [m]")
    ap.add_argument("--movable-post-thickness", type=float, default=None,
                    help="Son post lip thickness t_post [m]")
    ap.add_argument("--movable-center-gap-diameter", type=float, default=None,
                    help="fixed centre-gap outside diameter D_cg [m]")
    ap.add_argument("--movable-pintle-rod-diameter", type=float, default=None,
                    help="centre rod diameter D_pr [m]")
    ap.add_argument("--movable-maximum-opening", type=float, default=None,
                    help="physical open stop [m]; default derives a stop below "
                         "the Son centre-gap transition")
    ap.add_argument("--movable-commanded-opening", type=float, default=None,
                    help="fixed-mode mechanical opening L_open [m]")
    ap.add_argument("--movable-transition-area-fraction", type=float,
                    default=0.95,
                    help="derived open-stop A_tip/A_cg fraction (<1)")
    ap.add_argument("--movable-minimum-uniform-sheet-opening", type=float,
                    default=1.0e-4,
                    help="literature applicability warning floor for L_open [m]")
    ap.add_argument("--movable-cd-map", type=_parse_opening_cd_map,
                    default=(), metavar="F:CD,...",
                    help="configuration-controlled Cd versus L/Lmax map, e.g. "
                         "'0:0.62,0.5:0.70,1:0.76'")
    ap.add_argument("--movable-cd-source", default=None,
                    help="provenance label for --movable-cd-map")
    ap.add_argument("--movable-cd-sha256", default=None,
                    help="SHA-256 of the configuration-controlled Cd artifact")
    ap.add_argument("--movable-cd-geometry-sha256", default=None,
                    help="Son geometry fingerprint recorded by the Cd artifact")
    ap.add_argument("--movable-cd-fluid", default=None,
                    help="fluid identity covered by the Cd calibration")
    ap.add_argument("--movable-cd-re-min", type=float, default=None)
    ap.add_argument("--movable-cd-re-max", type=float, default=None)
    ap.add_argument("--movable-cd-dp-min", type=float, default=None)
    ap.add_argument("--movable-cd-dp-max", type=float, default=None)
    ap.add_argument("--movable-cd-temperature-min", type=float, default=None)
    ap.add_argument("--movable-cd-temperature-max", type=float, default=None)
    ap.add_argument("--movable-cd-cavitation-min", type=float, default=None)
    ap.add_argument("--movable-cd-cavitation-max", type=float, default=None)
    ap.add_argument("--movable-position-tolerance", type=float, default=None,
                    help="opening-position tolerance [m]")
    ap.add_argument("--movable-position-feedback-resolution", type=float,
                    default=None, help="position feedback resolution [m]")
    ap.add_argument("--movable-backlash", type=float, default=None,
                    help="bounded mechanism backlash [m]")
    ap.add_argument("--movable-metrology-source", default=None)
    ap.add_argument("--movable-metrology-sha256", default=None)
    ap.add_argument("--movable-closed-leakage-area", type=float, default=None,
                    help="measured/bounded closed-stop leakage area [m²]")
    ap.add_argument("--movable-leakage-source", default=None)
    ap.add_argument("--movable-leakage-sha256", default=None)
    ap.add_argument("--movable-unbalanced-pressure-area", type=float,
                    default=None,
                    help="explicit net projected pressure area [m²]; zero is "
                         "allowed only as a declared balanced assumption")
    ap.add_argument("--movable-spring-preload-force", type=float, default=0.0,
                    help="spring/preload force opposing motion [N]")
    ap.add_argument("--movable-seal-friction-force", type=float, default=None,
                    help="bounded seal/guide friction force [N]")
    ap.add_argument("--movable-moving-mass", type=float, default=None,
                    help="moving rod/actuator mass [kg]")
    ap.add_argument("--movable-maximum-acceleration", type=float, default=None,
                    help="declared maximum command acceleration [m/s²]")
    ap.add_argument("--movable-actuator-force-capacity", type=float,
                    default=None, help="available actuator force [N]")
    ap.add_argument("--movable-force-safety-factor", type=float, default=1.5)
    ap.add_argument("--movable-stem-diameter", type=float, default=None,
                    help="loaded actuator stem diameter [m]")
    ap.add_argument("--movable-stem-allowable-stress", type=float, default=None,
                    help="temperature-appropriate stem allowable [Pa]")
    ap.add_argument("--movable-actuator-source", default=None)
    ap.add_argument("--movable-actuator-sha256", default=None)
    ap.add_argument("--movable-sheet-thickness", type=float, default=None,
                    help="independently VOF-resolved or measured liquid-sheet "
                         "thickness [m], never L_open")
    ap.add_argument("--movable-sheet-thickness-method",
                    choices=("vof", "measured"), default=None)
    ap.add_argument("--movable-sheet-thickness-source", default=None)
    ap.add_argument("--movable-sheet-thickness-sha256", default=None)
    ap.add_argument("--movable-sheet-geometry-sha256", default=None,
                    help="Son geometry fingerprint recorded by the sheet artifact")
    ap.add_argument("--movable-sheet-thickness-fluid", default=None)
    ap.add_argument("--movable-sheet-opening-min", type=float, default=None)
    ap.add_argument("--movable-sheet-opening-max", type=float, default=None)
    ap.add_argument("--movable-sheet-dp-min", type=float, default=None)
    ap.add_argument("--movable-sheet-dp-max", type=float, default=None)
    ap.add_argument("--movable-sheet-mass-flow-min", type=float, default=None)
    ap.add_argument("--movable-sheet-mass-flow-max", type=float, default=None)
    ap.add_argument("--movable-axial-controller-dp-fraction-min", type=float,
                    default=1.0e-4)
    ap.add_argument("--movable-axial-controller-dp-fraction-max", type=float,
                    default=1.0)
    # fixed-geometry overrides (only used with --injector-sizing fixed)
    ap.add_argument("--pintle-annulus-gap", type=float, default=None)
    ap.add_argument("--pintle-slot-width", type=float, default=None)
    ap.add_argument("--pintle-slot-height", type=float, default=None)
    ap.add_argument("--pintle-slot-depth", type=float, default=None)
    ap.add_argument("--pintle-hole-diameter", type=float, default=None,
                    help="fixed-mode radial round-hole diameter [m]")
    ap.add_argument("--pintle-hole-length", type=float, default=None,
                    help="fixed-mode radial round-hole metering length [m]")
    ap.add_argument("--pintle-slot-corner-radius", type=float, default=None,
                    help="machined slot corner/end radius [m]")
    ap.add_argument("--pintle-slot-end-condition",
                    choices=("square", "rounded", "drilled", "edm"),
                    default="rounded",
                    help="machined slot end/corner condition")
    ap.add_argument("--pintle-tip-radius", type=float, default=None)
    ap.add_argument("--pintle-body-length", type=float, default=None)
    ap.add_argument("--pintle-annulus-length", type=float, default=None,
                    help="machined annular sleeve passage length [m]")
    ap.add_argument("--pintle-wall-thickness", type=float, default=None,
                    help="machined pintle post wall thickness [m]")
    ap.add_argument("--sleeve-wall-thickness", type=float, default=None,
                    help="machined annular sleeve wall thickness [m]")
    ap.add_argument("--injector-face-thickness", type=float, default=None)
    ap.add_argument("--injector-face-od", type=float, default=None)
    ap.add_argument("--fuel-inlet-count", type=int, default=2)
    ap.add_argument("--fuel-inlet-diameter", type=float, default=None,
                    help="fuel inlet/feed-port diameter [m]")
    ap.add_argument("--fuel-inlet-angle", type=float, default=0.0,
                    help="first fuel inlet angle on the face [deg]")
    ap.add_argument("--fuel-inlet-fitting", default="layout_only",
                    help="fuel fitting style label; layout_only is not a standard")
    ap.add_argument("--oxidizer-inlet-count", type=int, default=2)
    ap.add_argument("--oxidizer-inlet-diameter", type=float, default=None,
                    help="oxidizer inlet/feed-port diameter [m]")
    ap.add_argument("--oxidizer-inlet-angle", type=float, default=90.0,
                    help="first oxidizer inlet angle on the face [deg]")
    ap.add_argument("--oxidizer-inlet-fitting", default="layout_only",
                    help="oxidizer fitting style label; layout_only is not a standard")
    ap.add_argument("--fuel-manifold-width", type=float, default=None,
                    help="fuel annular manifold radial width [m]")
    ap.add_argument("--fuel-manifold-depth", type=float, default=None,
                    help="fuel annular manifold axial depth [m]")
    ap.add_argument("--oxidizer-manifold-width", type=float, default=None,
                    help="oxidizer annular manifold radial width [m]")
    ap.add_argument("--oxidizer-manifold-depth", type=float, default=None,
                    help="oxidizer annular manifold axial depth [m]")
    ap.add_argument("--manifold-velocity-limit", type=float, default=8.0,
                    help="manifold sizing velocity limit [m/s]")
    ap.add_argument("--inlet-velocity-limit", type=float, default=20.0,
                    help="inlet/feed-port sizing velocity limit [m/s]")
    ap.add_argument("--igniter-port-diameter", type=float, default=None,
                    help="central igniter port diameter [m]")
    ap.add_argument("--igniter-port-depth", type=float, default=None,
                    help="central igniter port modeled depth [m]")
    ap.add_argument("--seal-type", choices=("none", "o_ring", "gasket"),
                    default="o_ring")
    ap.add_argument("--o-ring-groove-width", type=float, default=None)
    ap.add_argument("--o-ring-groove-depth", type=float, default=None)
    ap.add_argument("--gasket-land-width", type=float, default=None)
    ap.add_argument("--min-tool-diameter", type=float, default=None,
                    help="minimum machining tool/EDM wire diameter [m]")
    ap.add_argument("--min-corner-radius", type=float, default=None,
                    help="minimum internal corner radius [m]")
    ap.add_argument("--injector-tolerance", type=float, default=None,
                    help="machined injector dimensional tolerance [m]")
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
                    help="render resolved MOC Mach/pressure/angle/temperature fields")
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
    ap.add_argument(
        "--require-brep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="require CadQuery/OpenCascade true B-rep STEP output (default); "
             "--no-require-brep permits an explicitly diagnostic faceted "
             "AP214 fallback that is not manufacturing CAD",
    )
    ap.add_argument("--regen-brep", action="store_true",
                    help="export a full-N one-solid regenerative wall STEP "
                         "(patterned positive ribs; requires CadQuery)")
    ap.add_argument("--regen-manifolds", action="store_true",
                    help="with --regen-brep, cut annular plenums and radial "
                         "ports into the one-solid wall")
    ap.add_argument(
        "--regen-release-mode",
        choices=("reference", "cold-flow"),
        default="reference",
        help=(
            "reference permits sealed-end channel geometry for CAD review; "
            "cold-flow requires --regen-manifolds and connected external ports"
        ),
    )
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
    ap.add_argument("--complete-package", action="store_true",
                    help="apply accessible starter defaults for a complete "
                         "nozzle + pintle injector + electric pump screening "
                         "package (bare runs do this automatically)")
    ap.add_argument("--tags", action="store_true",
                    help="print every run tag (flag) and exit")
    ap.add_argument("--no-banner", action="store_true", help="suppress the logo")
    ap.add_argument("--show", action="store_true",
                    help="pop up the flow-field / animation in a live window "
                         "(on by default for interactive runs)")
    ap.add_argument("--engine-mdo", action="store_true",
                    help="run the whole-engine differentiable MDO evaluation "
                         "(raosim.mdo.engine): nozzle + regen cooling + pintle "
                         "injector + electric pump feed solved as one coupled, "
                         "differentiable model; prints performance, the closed "
                         "cooling→feed hydraulic edge, mass ledger and all "
                         "constraint margins, then exits (uses --pc/--epsilon/"
                         "--target-thrust/--mixture-ratio/--*-injector-dp-"
                         "fraction/--pump-rpm)")
    ap.add_argument("--engine-mdo-couple-cstar", action="store_true",
                    help="with --engine-mdo/-optimize, enable the optional "
                         "spray→η_c* feedback edge (default frozen η_c*; this is "
                         "the RQ1 ablation knob — a screening correlation, not "
                         "validated physics)")
    ap.add_argument("--engine-mdo-optimize", action="store_true",
                    help="run the ε-constraint hard-constrained MDO "
                         "(raosim.mdo.nlp): minimise smooth electric-feed "
                         "objective mass s.t. "
                         "Isp ≥ --isp-min and every enforced discipline margin "
                         "≥ 0, solving for Pc/eps/injector-Δp/D_pintle/pump-rpm "
                         "with exact JAX Jacobians (SLSQP), while reporting "
                         "exact installed mass separately. Use --isp-sweep to "
                         "trace the mass–Isp Pareto frontier. Set the mission "
                         "with --target-thrust/--mixture-ratio/--burn-time/"
                         "--engine-mdo-ambient")
    ap.add_argument("--requirements", action="store_true",
                    help="requirement-driven design (raosim.requirements, "
                         "Layer 0): state performance TARGETS in NASA SP-125 "
                         "§2.1 terms and let the MDO choose Pc/eps/L*/channel "
                         "geometry/pintle/pump. Set the ask with "
                         "--target-thrust/--thrust-condition/--isp-min/"
                         "--flight-duration/--envelope-*-max/"
                         "--burnout-mass-max/--mdo-propellant. Every "
                         "requirement that is only partially screened, or not "
                         "screened at all, is reported as such")
    ap.add_argument("--thrust-condition", default="sea_level",
                    metavar="COND",
                    help="with --requirements: the back-pressure the thrust "
                         "target is quoted at — 'sea_level', 'vacuum', or "
                         "'altitude:<metres>'. SP-125 §2.1 quotes booster "
                         "thrust at sea level and upper-stage thrust in "
                         "vacuum, so this is part of the requirement, not "
                         "metadata (default: sea_level)")
    ap.add_argument("--isp-basis", choices=("thrust_chamber", "engine_system"),
                    default="thrust_chamber",
                    help="with --requirements: whether --isp-min refers to the "
                         "thrust chamber or the complete engine system "
                         "(SP-125 §2.1 requires this be stated). Only "
                         "thrust_chamber is screenable today")
    ap.add_argument("--flight-duration", type=float, default=None,
                    metavar="S",
                    help="with --requirements: rated flight duration [s]; "
                         "sizes the electric-feed energy (default: --burn-time)")
    ap.add_argument("--qualification-duration", type=float, default=None,
                    metavar="S",
                    help="with --requirements: cumulative demonstrated "
                         "duration [s]. SP-125 §2.1 says this governs most "
                         "design considerations, but no cumulative-life model "
                         "exists yet, so it is reported as unsupported")
    ap.add_argument("--envelope-diameter-max", type=float, default=None,
                    metavar="M",
                    help="with --requirements: maximum engine diameter [m] "
                         "(SP-125 §2.1 item 6). Screens the cooled chamber "
                         "only — the flange is host-side, so this is a lower "
                         "bound on the installed envelope")
    ap.add_argument("--envelope-length-max", type=float, default=None,
                    metavar="M",
                    help="with --requirements: maximum engine length [m], "
                         "injector face to nozzle exit")
    ap.add_argument("--burnout-mass-max", type=float, default=None,
                    metavar="KG",
                    help="with --requirements: engine mass at burnout [kg] "
                         "(SP-125 §2.1 item 5). Screens dry_mass_partial, a "
                         "lower bound: injector hardware, manifolds, valves, "
                         "lines, gimbal and mounts are not in it")
    ap.add_argument("--throttle-range", default=None, metavar="LO,HI",
                    help="with --requirements: required throttle range as a "
                         "thrust fraction (e.g. 0.6,1.0). Carried and "
                         "reported; not screenable at a single design point")
    ap.add_argument("--reusable-cycles", type=int, default=None, metavar="N",
                    help="with --requirements: required reuse cycles. Carried "
                         "and reported; structural_stress is a static screen, "
                         "not a cycle count")
    ap.add_argument("--isp-min", type=float, default=None,
                    help="with --engine-mdo-optimize: the Isp floor [s] "
                         "(ε-constraint) for a single min-mass solve")
    ap.add_argument("--isp-sweep", default=None, metavar="LO,HI,N",
                    help="with --engine-mdo-optimize: trace the Pareto frontier "
                         "over N Isp floors from LO to HI (e.g. 250,320,6)")
    ap.add_argument("--engine-mdo-ambient", type=float, default=None,
                    metavar="PA",
                    help="ambient pressure [Pa] for --engine-mdo/-optimize "
                         "(default sea level 101325; use a small value for the "
                         "altitude/vacuum frontier — the mass–Isp trade is only "
                         "strong when higher eps is not overexpanded)")
    ap.add_argument("--film-frac", type=float, default=None,
                    help="with --engine-mdo: fuel fraction diverted to wall film "
                         "cooling (0 = pure regen; reduces the coking wall temp "
                         "at a c* penalty).  Channel geometry reuses the existing "
                         "--channel-width / --channel-height flags.")
    ap.add_argument("--mdo-propellant", default=None, metavar="NAME",
                    help="with --engine-mdo/-optimize: propellant combination "
                         "(lox/rp-1, lox/lch4, lox/lh2, n2o4/mmh, n2o/ethanol). "
                         "Drives chamber gases, L*, densities, coolant "
                         "properties and the SP-8087 coolant wall limit.")
    ap.add_argument(
        "--mdo-chamber-property-table",
        type=Path,
        default=None,
        metavar="NPZ",
        help="validated sampled chamber-property table for MDO O/F/Pc physics; "
             "the file's content hash and stored propellant identity are checked",
    )
    ap.add_argument(
        "--mdo-of-mode",
        choices=("nominal", "pinned", "optimize"),
        default=None,
        help="mixture-ratio intent: nominal uses the propellant catalog, pinned "
             "requires --mixture-ratio, optimize requires an O/F-dependent "
             "--mdo-chamber-property-table",
    )
    ap.add_argument(
        "--mdo-mass-objective",
        choices=("min_dry_mass_partial", "min_electric_package_mass"),
        default="min_dry_mass_partial",
        help="explicit differentiable objective; partial dry mass includes the "
             "electric package, liner, channel lands, and closeout",
    )
    ap.add_argument(
        "--mdo-pc-search-min-pa",
        type=float,
        default=None,
        metavar="PA",
        help="override the recommended MDO chamber-pressure search-window lower endpoint",
    )
    ap.add_argument(
        "--mdo-pc-search-max-pa",
        type=float,
        default=None,
        metavar="PA",
        help="override the recommended MDO chamber-pressure search-window upper endpoint",
    )
    ap.add_argument(
        "--allow-incomplete-physics",
        action="store_true",
        help="permit an explicit screening run when governing physics coverage "
             "is unavailable; the physics verdict and exit status remain indeterminate",
    )
    ap.add_argument("--mdo-export", action="store_true",
                    help="with --engine-mdo/-optimize: hand the MDO design to "
                         "the authoritative LREKit pipeline (design_nozzle_v2) "
                         "for host reports and the Phase-11 discrepancy report. "
                         "The default is the established Bezier/TOP path; "
                         "--contour-method rao-bvp selects the preliminary "
                         "rao_variational_moc numerical analysis and requires "
                         "--cad none.")
    ap.add_argument("--design-margins", action="store_true",
                    help="with --engine-mdo/-optimize: apply the SP-8087 / "
                         "Mirzamoghadam hot-channel DESIGN MARGINS (+10%% heat "
                         "flux for injector streaking, -10%% channel flow for "
                         "maldistribution, 2x film-system capacity) instead of "
                         "nominal conditions")
    ap.add_argument("--t-wall", type=float, default=None, metavar="M",
                    help="with --engine-mdo: hot-gas wall thickness [m] "
                         "(default 8e-4; optimised in --engine-mdo-optimize)")
    ap.add_argument("--film-slot-height", type=float, default=None, metavar="M",
                    help="with --engine-mdo: tangential film-injector annular "
                         "slot height [m] (Hatch & Papell TN D-130 tested "
                         "0.0016-0.0127 m); sets the film injection velocity "
                         "via continuity")
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        expanded_argv = _expand_arg_files(raw_argv)
    except ValueError as exc:
        ap.error(str(exc))
    cli_argv = [sys.argv[0], *expanded_argv]

    args = ap.parse_args(expanded_argv)
    _reject_legacy_injector_pressure_drop(ap, cli_argv)
    bare = len(expanded_argv) == 0
    # Front door: a bare `lrekit` (or `lrekit -i`) asks which workflow to run
    # and then prompts for that workflow's parameters.  Explicit mode flags
    # skip the menu entirely, so scripted/CI use is unchanged.
    if (bare or args.interactive) and not (args.engine_mdo
                                           or args.engine_mdo_optimize):
        try:
            mode = _select_mode_interactively(args)
        except (EOFError, KeyboardInterrupt):
            print("\naborted")
            return 130
        if mode == "mdo":
            _validate_common_engine_args(args, ap)
            return _run_engine_mdo(args)
        if mode == "mdo_optimize":
            _validate_common_engine_args(args, ap)
            return _run_engine_mdo_optimize(args)
        bare = True          # traditional path keeps its starter defaults
        args.interactive = True
    _validate_common_engine_args(args, ap)
    if getattr(args, "requirements", False):
        return _run_requirements(args)
    if getattr(args, "engine_mdo_optimize", False):
        return _run_engine_mdo_optimize(args)
    if getattr(args, "engine_mdo", False):
        return _run_engine_mdo(args)
    if args.complete_package or bare:
        _apply_complete_package_defaults(
            args,
            cli_argv,
            reason="bare_run" if bare else "complete_package_flag",
        )
    _apply_wall_sizing_mode(args, ap, cli_argv)
    frozen_expansion_requested = (
        args.nozzle_expansion_model == "frozen-variable-cp"
    )
    if frozen_expansion_requested:
        if args.frozen_gas_table is None:
            ap.error(
                "--nozzle-expansion-model frozen-variable-cp requires "
                "--frozen-gas-table"
            )
        if args.contour_method != "bezier":
            ap.error(
                "frozen-variable-cp expansion is currently Bezier-only; "
                "constant-gamma MOC/Rao characteristic equations cannot "
                "accept a station-varying gamma"
            )
        if _argument_present(cli_argv, "--gamma"):
            ap.error(
                "--gamma cannot be supplied with frozen-variable-cp; gamma(T) "
                "is derived from the fixed-composition cp(T) table"
            )
        if args.cd_target is not None:
            ap.error(
                "--cd-target is unavailable with frozen-variable-cp because "
                "the Hall/SP-8120 throat-discharge screen is constant-gamma"
            )
        unsupported_thermal = []
        if args.regen:
            unsupported_thermal.append("--regen")
        if args.thermal:
            unsupported_thermal.append("--thermal")
        if args.auto_size:
            unsupported_thermal.append("--auto-size")
        if args.size_wall:
            unsupported_thermal.append("--size-wall")
        if unsupported_thermal:
            ap.error(
                "frozen-variable-cp currently blocks the constant-gamma "
                "Bartz/boundary-layer/regen sizing path; remove "
                + ", ".join(unsupported_thermal)
            )
    elif args.frozen_gas_table is not None:
        ap.error(
            "--frozen-gas-table requires --nozzle-expansion-model "
            "frozen-variable-cp"
        )
    if args.regen_manifolds:
        args.regen_brep = True
        args.hydraulic_network = True
    if args.regen_brep and not args.regen:
        ap.error("--regen-brep requires --regen")
    if args.regen_brep and args.cad == "none":
        ap.error("--regen-brep requires --cad step, ipt, or both")
    if args.regen_release_mode == "cold-flow" and not args.regen_manifolds:
        ap.error(
            "--regen-release-mode cold-flow requires --regen-manifolds so "
            "the exported channels have connected inlet/outlet ports"
        )
    if args.contour_method == "bezier" and (args.flowfield or args.animate):
        ap.error(
            "--flowfield/--animate require --contour-method rao-bvp; the "
            "Rao/TOP Bezier path has no resolved characteristic flow field"
        )
    if args.contour_method == "bezier" and args.allow_unconverged:
        ap.error("--allow-unconverged applies only to --contour-method rao-bvp")
    if args.contour_method != "bezier" and args.allow_chart_extrapolation:
        ap.error(
            "--allow-chart-extrapolation applies only to "
            "--contour-method bezier"
        )
    spray_assumptions_supplied = (
        args.spray_eta_mixing is not None
        or args.spray_eta_combustion is not None
    )
    if args.spray_cstar_coupling:
        if args.injector != "pintle":
            ap.error("--spray-cstar-coupling requires --injector pintle")
        from raosim.spray_coupling import SprayCStarCouplingSpec
        try:
            SprayCStarCouplingSpec(
                enabled=True,
                eta_mixing=args.spray_eta_mixing,
                eta_combustion=args.spray_eta_combustion,
                relaxation=args.spray_coupling_relaxation,
                relative_tolerance=args.spray_coupling_tolerance,
                max_iterations=args.spray_coupling_max_iterations,
            ).validate()
        except ValueError as exc:
            ap.error(str(exc))
    elif spray_assumptions_supplied:
        ap.error(
            "--spray-eta-mixing/--spray-eta-combustion require "
            "--spray-cstar-coupling"
        )
    if args.spray_evaporation_constant <= 0.0:
        ap.error("--spray-evaporation-constant must be positive")
    if args.require_release_evidence and args.release_evidence_manifest is None:
        ap.error(
            "--require-release-evidence requires --release-evidence-manifest"
        )
    if args.require_release_evidence and not str(args.configuration_id or "").strip():
        ap.error("--require-release-evidence requires --configuration-id")
    if args.gamma <= 1.0:
        ap.error("--gamma must be greater than one")
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
    shoulder_factor_explicit = _argument_present(cli_argv, "--shoulder-radius-factor")
    shoulder_sizing_explicit = _argument_present(cli_argv, "--shoulder-sizing")
    if (
        shoulder_factor_explicit
        and shoulder_sizing_explicit
        and args.shoulder_sizing == "auto"
    ):
        ap.error("--shoulder-radius-factor cannot be combined with --shoulder-sizing auto")
    if shoulder_factor_explicit and not shoulder_sizing_explicit:
        args.shoulder_sizing = "scalar"
    if args.shoulder_sizing == "scalar" and args.shoulder_radius_factor is None:
        args.shoulder_radius_factor = 0.25
        args._shoulder_radius_source = "legacy_scalar_placeholder"
    elif args.shoulder_radius_factor is not None:
        args._shoulder_radius_source = "user_supplied"
    else:
        args._shoulder_radius_source = "auto_pending"
    if args.shoulder_radius_factor is not None and args.shoulder_radius_factor <= 0.0:
        ap.error("--shoulder-radius-factor must be positive")
    if not 0.0 < args.shoulder_fill_fraction < 1.0:
        ap.error("--shoulder-fill-fraction must be in the open interval (0, 1)")
    ru_explicit = _argument_present(cli_argv, "--ru-factor")
    if args.cd_target is not None and ru_explicit:
        ap.error("--cd-target and --ru-factor both set the upstream throat radius")
    if args.cd_target is not None and not 0.0 < args.cd_target < 1.0:
        ap.error("--cd-target must be in (0, 1)")
    if args.ru_factor <= 0.0:
        ap.error("--ru-factor must be positive")
    if args.rd_factor <= 0.0:
        ap.error("--rd-factor must be positive")
    args._ru_factor_source = "user_supplied" if ru_explicit else "default"
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
    # Live windows for interactive/--show; saved files otherwise. Matplotlib is
    # imported lazily after parse-time exits so --help/--tags stay fast/quiet.
    show = _want_windows(args, expanded_argv)

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

    from raosim.release_readiness import (
        evaluate_release_readiness,
        load_evidence_manifest,
    )
    try:
        release_readiness = (
            load_evidence_manifest(
                args.release_evidence_manifest,
                expected_target="engine",
                expected_configuration_id=args.configuration_id,
            )
            if args.release_evidence_manifest is not None
            else evaluate_release_readiness(target="engine")
        )
        if args.require_release_evidence:
            release_readiness.require_complete()
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        ap.error(f"physical-release evidence rejected: {exc}")
    args._release_readiness = release_readiness

    # Ask for the inputs when run bare (no flags) or with -i/--interactive.
    if args.interactive or bare:
        if args.interactive and not getattr(args, "_complete_package_defaults", False):
            _apply_complete_package_defaults(
                args,
                cli_argv,
                reason="interactive_default",
            )
        print_tags()
        _interactive(args)
        _apply_wall_sizing_mode(args, ap, cli_argv)
        _validate_common_engine_args(args, ap)

    _ensure_pyplot(show)

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

    rt_explicit = _argument_present(cli_argv, "--rt")
    gamma_explicit = _argument_present(cli_argv, "--gamma")
    if gamma_explicit and args.thermo_mode == "cea":
        ap.error(
            "--gamma cannot override RocketCEA thermochemistry. Remove "
            "--gamma or select --thermo-mode constant-gamma."
        )
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
    if (
        args.spray_cstar_coupling
        and args.regen
        and not _coolant_is_cycle_fuel(args.coolant, fuel_name)
    ):
        ap.error(
            "--spray-cstar-coupling with --regen requires --coolant to be "
            "the cycle fuel; an independent coolant/bypass needs an explicit "
            "split and mixing model"
        )
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
    if gamma_explicit and prop_name is not None and args.gamma != prop.gamma:
        original_name = prop.name
        original_gamma = prop.gamma
        prop = custom_propellant(
            gamma=args.gamma,
            Mw=prop.Mw,
            Tc=prop.Tc,
            OF=prop.OF,
            eta_cstar=prop.eta_cstar,
            eta_CF=prop.eta_CF,
            source=(
                f"{prop.source}; user --gamma override replaced gamma="
                f"{original_gamma:g} with gamma={args.gamma:g}"
            ),
        )
        prop.name = original_name
        prop_warnings.append(
            f"Explicit --gamma={args.gamma:g} replaced the {original_name} "
            f"table value {original_gamma:g}; c*, performance, contour, "
            "separation, thermal, and injector calculations all use the "
            "overridden value."
        )
    frozen_gas = None
    if frozen_expansion_requested:
        if prop_name is None:
            ap.error(
                "frozen-variable-cp requires --propellant or an explicit "
                "--oxidizer/--fuel pair so the table can be checked against "
                "the selected chamber state"
            )
        try:
            from raosim.frozen_flow import load_frozen_gas_table

            frozen_gas = load_frozen_gas_table(args.frozen_gas_table)
            if (
                frozen_gas.freeze_basis == "chamber_equilibrium_snapshot"
                and not math.isclose(
                    float(frozen_gas.mixture_ratio),
                    float(args.mixture_ratio),
                    rel_tol=1.0e-9,
                    abs_tol=1.0e-12,
                )
            ):
                raise ValueError(
                    "table mixture_ratio does not match --mixture-ratio"
                )
            chamber_gamma = frozen_gas.gamma(prop.Tc)
        except (OSError, TypeError, ValueError) as exc:
            ap.error(f"frozen gas table rejected: {exc}")
        args.gamma = chamber_gamma
        prop_warnings.append(
            "Nozzle expansion uses the supplied thermally-perfect, "
            "fixed-composition cp(T) table. The scalar chamber gamma shown "
            "in geometry/injector diagnostics is derived from that table; "
            "MOC/Rao and thermal/regen paths are disabled."
        )
    else:
        # One authoritative value is used downstream.  This assignment also
        # covers the normal table/CEA path where --gamma was not supplied.
        args.gamma = prop.gamma
    args._frozen_gas = frozen_gas
    if args.cd_target is not None:
        try:
            radius_bounds = (
                REPOSITORY_UPSTREAM_RADIUS_RATIO_EXTENSION_BOUNDS
                if args.allow_throat_radius_extension
                else SP8120_UPSTREAM_RADIUS_RATIO_BOUNDS
            )
            args.ru_factor = upstream_radius_ratio_for_discharge_coefficient(
                args.cd_target,
                args.gamma,
                min_ratio=radius_bounds[0],
                max_ratio=radius_bounds[1],
            )
        except ValueError as exc:
            ap.error(str(exc))
        args._ru_factor_source = (
            "cd_target_hall_repository_extension"
            if args.ru_factor > SP8120_UPSTREAM_RADIUS_RATIO_BOUNDS[1]
            else "cd_target_hall_sp8120"
        )
    args._prop = prop
    args._prop_warnings = prop_warnings

    Pa = args.pa_over_p0 * args.pc
    try:
        if args.target_thrust is not None:
            args.rt = throat_radius_for_target_thrust(
                args.target_thrust,
                args.pc,
                Pa,
                args.epsilon,
                prop,
                frozen_gas=frozen_gas,
            )
        args._performance = compute_engine_performance(
            Pc=args.pc,
            Pa=Pa,
            Rt=args.rt,
            epsilon=args.epsilon,
            prop=prop,
            frozen_gas=frozen_gas,
        )
    except (TypeError, ValueError) as exc:
        ap.error(f"nozzle expansion rejected: {exc}")
    perf = args._performance
    args._mdot = perf.m_dot
    args._mdot_f = perf.m_dot / (1.0 + max(args.mixture_ratio, 0.0))
    args._mdot_o = args.mixture_ratio * args._mdot_f
    args._design_ambient = Pa

    # Injector pressure-drop fractions are authoritative. The legacy single
    # --injector-pressure-drop flag is rejected at parse time; any scalar dP
    # passed to the cooling solver below is derived from the fuel split only
    # when the regenerative coolant is the cycle fuel.
    _apply_split_injector_pressure_model(args, ap)
    _validate_pump_args(args, ap)

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
    shoulder_plan = (
        f"auto ({args.shoulder_fill_fraction:g}×max-feasible Rt)"
        if args.shoulder_sizing == "auto"
        else f"{args.shoulder_radius_factor:g} Rt"
    )
    cd_hall = throat_discharge_coefficient_hall(args.ru_factor, args.gamma)
    throat_plan = (
        f"Ru={args.ru_factor:g} Rt [{args._ru_factor_source}], "
        f"Rd={args.rd_factor:g} Rt, Cd_Hall={cd_hall:.4f}"
    )
    if args.cd_target is not None:
        throat_plan += f" (target {args.cd_target:g})"
    package_plan = (
        f"complete-package defaults [{args._complete_package_reason}: "
        + ", ".join(args._complete_package_default_notes)
        + "]"
        if getattr(args, "_complete_package_defaults", False)
        else "explicit flags / legacy defaults"
    )
    for k, v in [
        ("package", package_plan),
        ("nozzle", f"Rt={args.rt*1e3:g} mm, eps={args.epsilon:g}, "
                   f"L={args.length_pct:g}%, gamma={args.gamma:g}, "
                   f"method={args.contour_method}"),
        ("chamber", f"L*={args.l_star:g} m, CR={args.contraction_ratio:g}, "
                    f"shoulder={shoulder_plan}, "
                    f"Lc,min={args.minimum_cylindrical_length:g} m"),
        ("throat", throat_plan),
        ("solver", (
            f"{args.backend}  (max_nfev={args.max_nfev}, "
            f"n_control={args.n_control}, n_kernel={args.n_kernel})"
            if args.contour_method == "rao-bvp"
            else "not applicable (deterministic Rao/TOP chart geometry)"
        )),
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
        ("electric pump", (
            f"on  ({'auto' if args.pump_rpm is None else f'{args.pump_rpm:g}'} rpm, "
            f"{'auto' if args.motor_voltage is None else f'{args.motor_voltage:g}'} V, "
            f"burn {args.burn_time:g} s)"
            if args.electric_pump else dim("off"))),
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
    bvp_mode = args.contour_method == "rao-bvp"
    contour_label = (
        "Rao variational / MOC BVP"
        if bvp_mode else "Rao/TOP chart + quadratic Bezier"
    )
    print("\n" + cyan("▸ " + bold("Constructing contour")) +
          dim(f"  ({contour_label})"))
    if bvp_mode and args.backend == "jax" and args.max_nfev > 200:
        print(yellow("    note: full JAX LM solve — this runs for minutes."))
    sol = _solve(args)
    r = sol.residuals
    residual_tol = 2e-3
    gate = r.max_scaled <= residual_tol
    chart_extrapolated = bool(
        sol.construction_diagnostics.get("rao_chart_extrapolated", False)
    )
    chart_domain_gate = not chart_extrapolated
    da = sol.construction_diagnostics.get("design_angles", {})
    export_diag = sol.construction_diagnostics.get("export", {})
    if not isinstance(export_diag, dict):
        export_diag = {}
    wall_tangency_rms = sol.construction_diagnostics.get(
        "wall_tangency_rms", r.wall_tangency_rms
    )
    wall_tangency_rms_deg = (
        math.degrees(wall_tangency_rms)
        if wall_tangency_rms is not None else None
    )
    mass_gate = abs(r.mass_residual_rel) <= residual_tol
    length_gate = abs(r.length_residual_rel) <= residual_tol
    residual_gate = bool(gate and mass_gate and length_gate)
    endpoint_enforced = bool(
        export_diag.get("endpoint_enforced_for_export", False)
    )
    monotonic_cleanup = bool(
        export_diag.get("monotonic_cleanup_for_export", False)
    )
    postprocessed = bool(
        sol.construction_diagnostics.get("postprocessed", False)
    )
    no_postprocessing = not (
        postprocessed or endpoint_enforced or monotonic_cleanup
    )
    moc_compatibility_preserved = bool(
        sol.construction_diagnostics.get("moc_compatibility_preserved", False)
    )
    moc_gate = bool(
        no_postprocessing
        and moc_compatibility_preserved
        and wall_tangency_rms is not None
        and wall_tangency_rms < math.radians(0.25)
        and r.characteristic_crossings == 0
    )
    boundary_min = sol.construction_diagnostics.get("boundary_min")
    valid_region_gate = bool(
        boundary_min is not None and boundary_min >= -residual_tol
    )
    thrust_sanity = sol.construction_diagnostics.get("thrust_sanity")
    if not isinstance(thrust_sanity, dict):
        thrust_sanity = {}
    thrust_sanity_applicable = bool(thrust_sanity.get("applicable", True))
    thrust_sanity_gate = bool(thrust_sanity.get("passes", False))
    net_report = sol.construction_diagnostics.get("net_report")
    if not isinstance(net_report, dict):
        net_report = {}
    crossing_samples = net_report.get("crossing_samples")
    if not isinstance(crossing_samples, list):
        crossing_samples = []
    promotion_blockers: list[str] = []
    if not bvp_mode and not chart_domain_gate:
        promotion_blockers.append("Rao/TOP chart extrapolation")
    if bvp_mode and args.max_nfev <= 0:
        promotion_blockers.append("seed-only contour")
    elif bvp_mode and not residual_gate:
        promotion_blockers.append("BVP residual closure")
    if bvp_mode and not moc_gate:
        moc_notes: list[str] = []
        if not moc_compatibility_preserved:
            if str(net_report.get("audit_basis", "")).startswith("bde_"):
                if not net_report.get("bde_physical_mesh_complete", True):
                    moc_notes.append("physical B-D-E mesh incomplete")
                n_trunc = int(net_report.get(
                    "bde_negative_r_truncated_rows", 0
                ) or 0)
                if n_trunc:
                    moc_notes.append(
                        f"{n_trunc} negative-r truncated rows"
                    )
                if not net_report.get("measured_crossing_passes", True):
                    moc_notes.append("measured mesh crossings")
                if not net_report.get("measured_cell_orientation_passes", True):
                    moc_notes.append(
                        f"{int(net_report.get('measured_invalid_cell_count', 0))} "
                        "zero/negative-area cells"
                    )
                if not net_report.get("measured_compatibility_passes", True):
                    moc_notes.append(
                        "BDE compatibility "
                        f"C-={float(net_report.get('cminus_max', float('nan'))):.3g} deg, "
                        f"C+={float(net_report.get('cplus_max', float('nan'))):.3g} deg"
                    )
                if not net_report.get("measured_mach_line_direction_passes", True):
                    moc_notes.append("Mach-line direction mismatch")
                if not net_report.get("axis_condition_passes", True):
                    moc_notes.append("axis regularity")
                if not net_report.get("measured_neighbor_smoothness_passes", True):
                    moc_notes.append("neighbor-state smoothness")
                if not net_report.get("axial_mass_conservation_passes", True):
                    mass_error = net_report.get("axial_mass_cut_max_rel_error")
                    if isinstance(mass_error, (int, float)):
                        moc_notes.append(
                            f"axial mass error {100.0 * mass_error:.2f}%"
                        )
                    else:
                        moc_notes.append("axial mass conservation")
                if not net_report.get("measured_wall_tangency_passes", True):
                    wall_tmax = net_report.get("wall_tangency_max_deg")
                    if isinstance(wall_tmax, (int, float)):
                        moc_notes.append(
                            f"BDE wall tangency max {wall_tmax:.2f} deg"
                        )
                    else:
                        moc_notes.append("BDE wall tangency")
            else:
                moc_notes.append("net compatibility")
        if wall_tangency_rms is None:
            moc_notes.append("wall tangency unavailable")
        elif wall_tangency_rms >= math.radians(0.25):
            moc_notes.append(
                f"wall tangency {math.degrees(wall_tangency_rms):.2f} deg"
            )
        if r.characteristic_crossings:
            moc_notes.append(f"{r.characteristic_crossings} crossings")
        if not no_postprocessing:
            moc_notes.append("export postprocessing")
        promotion_blockers.append(
            "MOC closure" + (
                f" ({', '.join(moc_notes)})" if moc_notes else ""
            )
        )
    if bvp_mode and not valid_region_gate:
        promotion_blockers.append(f"Rao valid region boundary={boundary_min}")
    if bvp_mode and not thrust_sanity_gate:
        if not thrust_sanity_applicable:
            mass_fraction = thrust_sanity.get("kernel_bd_mass_fraction")
            scaled_error = thrust_sanity.get("mass_fraction_scaled_cf_rel_error")
            thrust_notes: list[str] = []
            if (
                isinstance(mass_fraction, (int, float))
                and math.isfinite(mass_fraction)
            ):
                thrust_notes.append(
                    f"DE mass fraction {100.0 * mass_fraction:.1f}%"
                )
            if (
                isinstance(scaled_error, (int, float))
                and math.isfinite(scaled_error)
            ):
                thrust_notes.append(
                    f"mass-scaled Cf error {100.0 * scaled_error:.1f}%"
                )
            promotion_blockers.append(
                "full-control-surface thrust audit unavailable"
                + (f" ({', '.join(thrust_notes)})" if thrust_notes else "")
            )
        elif (
            isinstance((cf_rel_error := thrust_sanity.get("cf_rel_error")),
                       (int, float))
            and math.isfinite(cf_rel_error)
        ):
            promotion_blockers.append(
                f"thrust sanity Cf error {100.0 * cf_rel_error:.1f}%"
            )
        else:
            promotion_blockers.append("thrust sanity")
    contour_reliability = {
        "solver_backend": args.backend,
        "contour_method": args.contour_method,
        "solver_gates_applicable": bvp_mode,
        "rao_chart_domain": sol.construction_diagnostics.get(
            "rao_chart_domain"
        ),
        "rao_chart_extrapolated": chart_extrapolated,
        "rao_chart_domain_gate_passed": chart_domain_gate,
        "max_nfev": int(args.max_nfev),
        "seed_only": bool(args.max_nfev <= 0),
        "residual_tol": residual_tol,
        "max_scaled": float(r.max_scaled),
        "rms_scaled": float(r.rms_scaled),
        "mass_residual_rel": float(r.mass_residual_rel),
        "length_residual_rel": float(r.length_residual_rel),
        "max_scaled_gate_passed": bool(gate),
        "mass_residual_gate_passed": bool(mass_gate),
        "length_residual_gate_passed": bool(length_gate),
        "residual_gate_passed": bool(residual_gate),
        "moc_gate_passed": bool(moc_gate),
        "valid_region_gate_passed": bool(valid_region_gate),
        "thrust_sanity_gate_applicable": bool(thrust_sanity_applicable),
        "thrust_sanity_gate_passed": bool(thrust_sanity_gate),
        "optimization_converged": bool(sol.converged),
        "reliability": sol.reliability.value,
        "promotion_blockers": promotion_blockers,
        "moc_compatibility_preserved": moc_compatibility_preserved,
        "wall_tangency_rms_deg": wall_tangency_rms_deg,
        "characteristic_crossings": int(r.characteristic_crossings),
        "characteristic_crossing_samples": crossing_samples,
        "bde_integrity": {
            key: net_report.get(key) for key in (
                "measured_mesh_source",
                "measured_mesh_link_count",
                "crossings",
                "cminus_rms",
                "cminus_max",
                "cplus_rms",
                "cplus_max",
                "compatibility_tol_deg",
                "measured_compatibility_passes",
                "mach_line_direction_rms_deg",
                "mach_line_direction_max_deg",
                "mach_line_direction_tol_deg",
                "measured_mach_line_direction_passes",
                "measured_cell_count",
                "measured_invalid_cell_count",
                "measured_min_oriented_cell_area_m2",
                "measured_cell_orientation_passes",
                "neighbor_mach_p99",
                "neighbor_mach_max",
                "neighbor_theta_p99_deg",
                "neighbor_theta_max_deg",
                "neighbor_pressure_ratio_p99",
                "neighbor_pressure_ratio_max",
                "measured_neighbor_smoothness_passes",
                "axis_node_count",
                "axis_max_abs_r_m",
                "axis_max_abs_theta_deg",
                "axis_mach_min",
                "axis_mach_max",
                "axis_thermodynamics_finite",
                "axis_condition_passes",
                "axial_mass_cut_count",
                "axial_mass_cut_valid_count",
                "axial_mass_cut_x_m",
                "axial_mass_cut_coverage",
                "axial_mass_cut_rel_errors",
                "axial_mass_cut_max_rel_error",
                "axial_mass_cut_rel_tol",
                "axial_mass_conservation_passes",
                "bde_physical_mesh_complete",
                "bde_auxiliary_continuation_caustic",
                "bde_auxiliary_frontier_min_x_m",
                "bde_auxiliary_caustic_downstream_of_exit",
            )
        },
        "postprocessed": postprocessed,
        "endpoint_enforced_for_export": endpoint_enforced,
        "monotonic_cleanup_for_export": monotonic_cleanup,
        "wall_export_interpolation_basis": export_diag.get(
            "interpolation_basis"
        ),
        "wall_export_point_count": export_diag.get("export_point_count"),
        "wall_export_max_adjacent_turn_deg": export_diag.get(
            "max_adjacent_turn_deg"
        ),
        "wall_export_max_adjacent_turn_tol_deg": export_diag.get(
            "max_adjacent_turn_tol_deg"
        ),
        "wall_export_turn_gate_passed": export_diag.get(
            "polyline_turn_gate_passed"
        ),
        "rao_region": sol.construction_diagnostics.get("rao_region"),
        "boundary_min": boundary_min,
        "thrust_sanity": thrust_sanity,
        "warnings": list(sol.warnings),
        "quasi_1d_expansion_model": perf.expansion_model,
        "frozen_property_performance_benchmark_passed": (
            None if perf.frozen_flow is None else False
        ),
        "variable_property_moc_rao_applicable": (
            None if perf.frozen_flow is None else False
        ),
    }
    badge = (
        (green("✓ trusted chart path") if chart_domain_gate
         else yellow("● unqualified chart extrapolation"))
        if not bvp_mode
        else green("✓ gate passed") if gate
        else yellow("● seed / not converged")
    )
    print(f"    max_scaled={r.max_scaled:.3e}   {badge}   "
          f"reliability={sol.reliability.value}")
    if bvp_mode and args.max_nfev <= 0:
        print(yellow("    seed-only contour: --max-nfev 0 skipped the "
                     "least-squares/JAX solve; no residual-solved promotion."))
    elif bvp_mode and not gate:
        print(yellow("    residual gate failed: "
                     f"max_scaled {r.max_scaled:.3e} > {residual_tol:.1e}."))
    elif sol.reliability.value == "geometric_approximation" and promotion_blockers:
        print(yellow("    reliability blockers: "
                     + "; ".join(promotion_blockers)))
    if crossing_samples:
        first_crossing = crossing_samples[0]
        s1 = first_crossing.get("segment_1", {}) or {}
        s2 = first_crossing.get("segment_2", {}) or {}
        ix = first_crossing.get("intersection", {}) or {}
        loc = (
            f"x={1000.0 * ix.get('x', 0.0):.2f} mm, "
            f"r={1000.0 * ix.get('r', 0.0):.2f} mm"
            if ix else "location unavailable"
        )
        print(yellow(
            "    first MOC crossing sample: "
            f"{s1.get('family', '?')} row {s1.get('row', '?')} vs "
            f"{s2.get('family', '?')} row {s2.get('row', '?')} at {loc}"
        ))
    if bvp_mode and thrust_sanity_gate and not thrust_sanity_applicable:
        mass_fraction = thrust_sanity.get("kernel_bd_mass_fraction")
        scaled_error = thrust_sanity.get("mass_fraction_scaled_cf_rel_error")
        notes: list[str] = []
        if (
            isinstance(mass_fraction, (int, float))
            and math.isfinite(mass_fraction)
        ):
            notes.append(f"DE mass fraction {100.0 * mass_fraction:.1f}%")
        if (
            isinstance(scaled_error, (int, float))
            and math.isfinite(scaled_error)
        ):
            notes.append(f"mass-scaled Cf error {100.0 * scaled_error:.1f}%")
        print(dim(
            "    thrust consistency: "
            + (", ".join(notes) if notes else "partial D-E diagnostic only")
            + " (full kernel-connected C-D-E reconstruction unavailable for this solve)"
        ))
    print(f"    theta_N={math.degrees(sol.theta_N):.2f}° "
          f"{dim('['+da.get('theta_N_source','?')+']')}   "
          f"theta_E={math.degrees(sol.theta_E):.2f}°   "
          f"Cf={bold('%.4f' % sol.thrust_coefficient)}")
    if export_diag.get("interpolation_basis"):
        print(dim(
            "    wall export: "
            f"{export_diag['interpolation_basis']}, "
            f"{int(export_diag.get('export_point_count', 0))} points, "
            "max adjacent turn "
            f"{float(export_diag.get('max_adjacent_turn_deg', float('nan'))):.3f}°"
        ))
    contour_export_gate = bool(
        chart_domain_gate
        if not bvp_mode
        else (
            sol.converged
            and residual_gate
            and moc_gate
            and valid_region_gate
            and thrust_sanity_gate
        )
    )
    diagnostic_export_override = bool(
        (bvp_mode and args.allow_unconverged)
        or (not bvp_mode and args.allow_chart_extrapolation)
    )
    if not contour_export_gate and not diagnostic_export_override:
        print(red(
            "    contour failed the export gate; no contour "
            "or CAD artifacts were written.  Blockers: "
            + "; ".join(promotion_blockers)
            + (
                ". Rerun with --allow-unconverged only for explicit "
                "diagnostic artifacts."
                if bvp_mode else
                ". Rerun with --allow-chart-extrapolation only for explicit "
                "diagnostic artifacts."
            )
        ), flush=True)
        return 2
    if not contour_export_gate:
        print(yellow(
            "    diagnostic override active: exporting a contour with "
            "unresolved reliability blockers."
        ))

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
    nozzle["quasi_1d_expansion_model"] = perf.expansion_model
    nozzle["frozen_flow_fingerprint_sha256"] = (
        perf.frozen_flow_fingerprint
    )
    nozzle["throat_geometry"] = throat_geometry.to_dict()
    nozzle["throat_location"] = throat_geometry.throat_location
    try:
        if args.shoulder_sizing == "auto":
            args.shoulder_radius_factor = auto_shoulder_factor(
                args.rt,
                args.contraction_ratio,
                throat_geometry=throat_geometry,
                fill_fraction=args.shoulder_fill_fraction,
            )
            args._shoulder_radius_source = "auto_geometric_closure"
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
        "backend": args.backend, "contour_method": args.contour_method,
        "Rt": args.rt, "epsilon": args.epsilon,
        "length_pct": args.length_pct, "gamma": args.gamma,
        "run_defaults": {
            "complete_package": getattr(args, "_complete_package_defaults", False),
            "reason": getattr(args, "_complete_package_reason", None),
            "notes": getattr(args, "_complete_package_default_notes", []),
        },
        "max_scaled": float(r.max_scaled), "gate_2e3": bool(gate),
        "reliability": sol.reliability.value,
        "contour_reliability": contour_reliability,
        "theta_N_deg": math.degrees(sol.theta_N),
        "theta_E_deg": math.degrees(sol.theta_E),
        "Cf": float(sol.thrust_coefficient),
        "thrust_closure": {
            "sizing_basis": (
                "quasi_1d_thermally_perfect_frozen_composition"
                if perf.frozen_flow is not None
                else "quasi_1d_calorically_perfect_attached_flow"
            ),
            "quasi_1d_Cf_ideal": float(perf.Cf_ideal),
            "quasi_1d_Cf_delivered": float(perf.Cf_actual),
            "contour_audit_Cf": float(sol.thrust_coefficient),
            "contour_audit_basis": (
                "same_quasi_1d_model"
                if not bvp_mode else thrust_sanity.get(
                    "gate_basis", "unavailable_full_cde_surface"
                )
            ),
            "relative_difference_from_quasi_1d_ideal": float(
                (sol.thrust_coefficient - perf.Cf_ideal)
                / max(abs(perf.Cf_ideal), 1e-30)
            ),
            "export_gate_passed": contour_export_gate,
            "diagnostic_override": bool(
                not contour_export_gate and diagnostic_export_override
            ),
        },
        "exit_radius": float(y[-1]),
        "throat_geometry": {
            "upstream_radius_ratio": args.ru_factor,
            "upstream_radius_source": args._ru_factor_source,
            "discharge_coefficient_hall": cd_hall,
            "discharge_coefficient_target": args.cd_target,
            "discharge_coefficient_model_applicable": (
                perf.frozen_flow is None
            ),
            "downstream_radius_ratio": args.rd_factor,
            "convergent_half_angle_deg": args.convergent_angle,
        },
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
            "injector_face_to_throat_length_m":
                chamber["injector_to_throat_length"],
            "target_volume_m3": chamber["V_target"],
            "polyline_frustum_volume_m3": chamber["V_chamber"],
            "geometry_checks": contour["geometry_checks"],
        },
    }
    from raosim.model_registry import audit_model_registry, model_provenance_dict
    summary["hardware_qualified"] = False
    summary["configuration_id"] = args.configuration_id
    summary["physical_release_readiness"] = (
        args._release_readiness.to_dict()
    )
    summary["model_registry_audit"] = audit_model_registry(REPO_ROOT)
    summary["model_provenance"] = model_provenance_dict()
    summary["spray_cstar_coupling"] = {
        "enabled": False,
        "status": "disabled",
        "reason": "requires an explicitly requested pintle injector and efficiencies",
    }
    summary["flow_model_gates"] = {
        "frozen_q1d_conservation_closure": (
            None
            if perf.frozen_flow is None
            else bool(perf.frozen_flow.all_closures_pass)
        ),
        "frozen_property_and_performance_benchmark": (
            None if perf.frozen_flow is None else False
        ),
        "variable_property_moc_rao": (
            None if perf.frozen_flow is None else False
        ),
        "variable_property_bartz_boundary_layer_regen": (
            None if perf.frozen_flow is None else False
        ),
        "variable_property_hall_throat_cd": (
            None if perf.frozen_flow is None else False
        ),
    }
    summary["performance"] = {
        "propellant": perf.propellant_name,
        "thermo_mode": args.thermo_mode,
        "source": prop.source,
        "Pc_pa": args.pc,
        "Pa_pa": Pa,
        "mixture_ratio": args.mixture_ratio,
        "gamma": perf.gamma,
        "gamma_throat": perf.gamma_throat,
        "gamma_exit": perf.gamma_exit,
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
        "exit_temperature_K": perf.exit_temperature,
        "expansion_model": perf.expansion_model,
        "frozen_flow_fingerprint_sha256": perf.frozen_flow_fingerprint,
        "frozen_expansion": (
            perf.frozen_flow.as_dict() if perf.frozen_flow is not None else None
        ),
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
    if crossing_samples:
        _write_moc_crossing_samples(
            args.out / "contour_moc_crossings.csv",
            crossing_samples,
        )
        artifacts.append("contour_moc_crossings.csv")
        print(yellow(
            "    wrote contour_moc_crossings.csv "
            f"({len(crossing_samples)} sampled crossings)"
        ))

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
            frozen_expansion=perf.frozen_flow,
            save_path=args.out / "separation_contour.png", show=show)
        fig.clf()
        artifacts.append("separation_contour.png")
        print(green("    wrote separation_contour.png"))
    except Exception as exc:  # plotting must never break the run
        print(yellow(f"    separation_contour.png skipped: {exc}"))

    # ---- MOC / Rao construction diagrams -----------------------------
    # The characteristic contour is built by the NASA/JHU MOC kernel + Rao
    # B-D-E topology + BDE remaining-mesh march; these render that actual
    # construction (kernel expansion fan, B-D-E topology, BDE net) from the
    # in-memory artifacts the bde wall method stashes on the solution.  They
    # only exist for an evaluate_moc bde solve, so each is guarded and skips
    # cleanly otherwise (plotting must never break the run).
    if sol.construction_diagnostics.get("bde_artifacts"):
        print("\n" + cyan("▸ " + bold("MOC construction")) +
              dim("  (kernel fan, Rao topology, BDE net + diagnostics)"))
        for _fn, _name, _label in (
            ("plot_kernel_expansion_fan", "kernel_fan.png",
             "throat kernel / expansion fan"),
            ("plot_rao_topology", "rao_topology.png",
             "Rao B-D-E topology"),
            ("plot_bde_mesh", "bde_mesh.png",
             "BDE characteristic net"),
            ("plot_bde_diagnostics", "bde_diagnostics.png",
             "BDE flow / compatibility / mass diagnostics"),
            ("plot_bde_integrity", "bde_integrity.png",
             "BDE link / cell / axis / smoothness integrity audit"),
        ):
            try:
                import raosim.moc_diagrams as _md
                _kwargs = {}
                if _fn == "plot_bde_diagnostics":
                    _kwargs = {"gamma": args.gamma, "p0": args.pc}
                elif _fn == "plot_bde_integrity":
                    _kwargs = {"gamma": args.gamma, "residual_tol": 2e-3}
                fig = getattr(_md, _fn)(
                    sol, save_path=args.out / _name, show=show, **_kwargs
                )
                if not show:
                    fig.clf()
                artifacts.append(_name)
                print(green(f"    wrote {_name}")
                      + (dim("  (window)") if show else ""))
            except Exception as exc:  # plotting must never break the run
                print(yellow(f"    {_name} skipped: {exc}"))

    # ---- optional steady flow-field render ---------------------------
    if args.flowfield:
        print("\n" + cyan("▸ " + bold("Flow field")) +
              dim("  (resolved MOC Mach/p/theta/T + characteristics)"))
        try:
            from raosim.flow_viz import plot_flowfield
            fig = plot_flowfield(sol, gamma=args.gamma, Tc=args.chamber_temp,
                                 exit_mach=perf.Me,
                                 save_path=args.out / "flowfield.png", show=show)
            field_coverage = getattr(fig, "flowfield_coverage", {})
            if not show:
                fig.clf()
            artifacts.append("flowfield.png")
            print(green("    wrote flowfield.png")
                  + (dim("  (window)") if show else ""))
            if field_coverage.get("full_field"):
                print(dim(
                    "    resolved wall-to-axis field: "
                    f"x={1e3 * float(field_coverage['resolved_x_min_m']):.1f} "
                    "mm to exit; exit radial coverage "
                    f"{100.0 * float(field_coverage['exit_radial_fraction']):.1f}%"
                ))
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
            fuel_injector_dp_fraction=args._regen_fuel_injector_dp_fraction,
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
            fuel_injector_dp_fraction=args._regen_fuel_injector_dp_fraction,
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
            injector_pressure_drop=args._regen_injector_pressure_drop,
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
            injector_pressure_drop=args._regen_injector_pressure_drop)
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
            PintleMechanicalSpec, InjectorManufacturingSpec, InjectorUnsupportedState,
            InjectorSpecError, MovablePintleSpec, evaluate_pintle_injector,
        )
        from raosim.design import (
            CoolingSpec,
            MaterialSpec,
            SprayRegenIterationPayload,
        )
        print("\n" + cyan("▸ " + bold("Injector")) +
              dim("  (pintle, liquid/liquid, sized from the ṁ split)"))
        if not args._ox_name or not args._fuel_name:
            print(red("    --injector pintle needs real propellant identities; "
                      "pass --oxidizer/--fuel or --propellant 'OX/FUEL'."))
            return 2
        range_inputs = (
            ("movable Cd Reynolds", args.movable_cd_re_min, args.movable_cd_re_max),
            ("movable Cd pressure-drop", args.movable_cd_dp_min, args.movable_cd_dp_max),
            (
                "movable Cd temperature",
                args.movable_cd_temperature_min,
                args.movable_cd_temperature_max,
            ),
            (
                "movable Cd cavitation-number",
                args.movable_cd_cavitation_min,
                args.movable_cd_cavitation_max,
            ),
            (
                "movable sheet opening",
                args.movable_sheet_opening_min,
                args.movable_sheet_opening_max,
            ),
            (
                "movable sheet pressure-drop",
                args.movable_sheet_dp_min,
                args.movable_sheet_dp_max,
            ),
            (
                "movable sheet mass-flow",
                args.movable_sheet_mass_flow_min,
                args.movable_sheet_mass_flow_max,
            ),
        )
        for label, lower, upper in range_inputs:
            if (lower is None) != (upper is None):
                print(red(
                    f"    {label} validity requires both minimum and maximum."
                ))
                return 2

        def optional_range(lower, upper):
            return None if lower is None else (lower, upper)

        movable_cd_reynolds_range = optional_range(
            args.movable_cd_re_min, args.movable_cd_re_max
        )
        movable_cd_pressure_drop_range = optional_range(
            args.movable_cd_dp_min, args.movable_cd_dp_max
        )
        movable_cd_temperature_range = optional_range(
            args.movable_cd_temperature_min, args.movable_cd_temperature_max
        )
        movable_cd_cavitation_range = optional_range(
            args.movable_cd_cavitation_min, args.movable_cd_cavitation_max
        )
        movable_sheet_opening_range = optional_range(
            args.movable_sheet_opening_min, args.movable_sheet_opening_max
        )
        movable_sheet_dp_range = optional_range(
            args.movable_sheet_dp_min, args.movable_sheet_dp_max
        )
        movable_sheet_mass_flow_range = optional_range(
            args.movable_sheet_mass_flow_min,
            args.movable_sheet_mass_flow_max,
        )
        inj_spec = InjectorSpec(
            type="pintle", architecture=args.injector_architecture,
            sizing=args.injector_sizing,
            fuel_dp_fraction=args.fuel_injector_dp_fraction,
            oxidizer_dp_fraction=args.oxidizer_injector_dp_fraction,
            fuel_cd=args.fuel_discharge_coefficient,
            oxidizer_cd=args.oxidizer_discharge_coefficient,
            faceplate_material=args.material, pintle_material=args.material,
            target_momentum_ratio=args.pintle_target_momentum_ratio,
            movable_axial_controller_dp_fraction_bounds=(
                args.movable_axial_controller_dp_fraction_min,
                args.movable_axial_controller_dp_fraction_max,
            ),
            evaporation_constant=args.spray_evaporation_constant,
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
                radial_exit_style=args.pintle_radial_exit,
                radial_hole_diameter=args.pintle_hole_diameter,
                radial_hole_length=args.pintle_hole_length,
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
            mechanical=PintleMechanicalSpec(
                bolt_count=args.bolt_count,
                bolt_circle_diameter=args.bolt_circle,
                bolt_hole_diameter=args.bolt_hole,
                faceplate_thickness=args.injector_face_thickness,
                faceplate_outer_diameter=args.injector_face_od,
                fuel_inlet_count=args.fuel_inlet_count,
                fuel_inlet_diameter=args.fuel_inlet_diameter,
                fuel_inlet_angle=args.fuel_inlet_angle,
                fuel_inlet_fitting=args.fuel_inlet_fitting,
                oxidizer_inlet_count=args.oxidizer_inlet_count,
                oxidizer_inlet_diameter=args.oxidizer_inlet_diameter,
                oxidizer_inlet_angle=args.oxidizer_inlet_angle,
                oxidizer_inlet_fitting=args.oxidizer_inlet_fitting,
                fuel_manifold_width=args.fuel_manifold_width,
                fuel_manifold_depth=args.fuel_manifold_depth,
                oxidizer_manifold_width=args.oxidizer_manifold_width,
                oxidizer_manifold_depth=args.oxidizer_manifold_depth,
                manifold_velocity_limit=args.manifold_velocity_limit,
                inlet_velocity_limit=args.inlet_velocity_limit,
                slot_depth=args.pintle_slot_depth,
                slot_corner_radius=args.pintle_slot_corner_radius,
                slot_end_condition=args.pintle_slot_end_condition,
                annulus_length=args.pintle_annulus_length,
                sleeve_wall_thickness=args.sleeve_wall_thickness,
                pintle_wall_thickness=args.pintle_wall_thickness,
                igniter_port_diameter=args.igniter_port_diameter,
                igniter_port_depth=args.igniter_port_depth,
                seal_type=args.seal_type,
                o_ring_groove_width=args.o_ring_groove_width,
                o_ring_groove_depth=args.o_ring_groove_depth,
                gasket_land_width=args.gasket_land_width,
                min_tool_diameter=args.min_tool_diameter,
                min_corner_radius=args.min_corner_radius,
                tolerance=args.injector_tolerance),
            movable_pintle=MovablePintleSpec(
                post_diameter=args.movable_post_diameter,
                post_thickness=args.movable_post_thickness,
                center_gap_diameter=args.movable_center_gap_diameter,
                pintle_rod_diameter=args.movable_pintle_rod_diameter,
                maximum_opening=args.movable_maximum_opening,
                commanded_opening=args.movable_commanded_opening,
                transition_area_fraction=(
                    args.movable_transition_area_fraction
                ),
                minimum_uniform_sheet_opening=(
                    args.movable_minimum_uniform_sheet_opening
                ),
                cd_vs_opening_fraction=args.movable_cd_map,
                cd_calibration_source=args.movable_cd_source,
                cd_calibration_artifact_sha256=args.movable_cd_sha256,
                cd_geometry_fingerprint_sha256=(
                    args.movable_cd_geometry_sha256
                ),
                cd_reynolds_range=movable_cd_reynolds_range,
                cd_pressure_drop_range=movable_cd_pressure_drop_range,
                cd_temperature_range=movable_cd_temperature_range,
                cd_cavitation_number_range=movable_cd_cavitation_range,
                cd_fluid_name=args.movable_cd_fluid,
                position_tolerance=args.movable_position_tolerance,
                position_feedback_resolution=(
                    args.movable_position_feedback_resolution
                ),
                backlash=args.movable_backlash,
                closed_leakage_area=args.movable_closed_leakage_area,
                metrology_source=args.movable_metrology_source,
                metrology_artifact_sha256=args.movable_metrology_sha256,
                leakage_source=args.movable_leakage_source,
                leakage_artifact_sha256=args.movable_leakage_sha256,
                unbalanced_pressure_area=(
                    args.movable_unbalanced_pressure_area
                ),
                spring_preload_force=args.movable_spring_preload_force,
                seal_friction_force=args.movable_seal_friction_force,
                moving_mass=args.movable_moving_mass,
                maximum_acceleration=args.movable_maximum_acceleration,
                actuator_force_capacity=(
                    args.movable_actuator_force_capacity
                ),
                force_safety_factor=args.movable_force_safety_factor,
                stem_diameter=args.movable_stem_diameter,
                stem_allowable_stress=args.movable_stem_allowable_stress,
                actuator_source=args.movable_actuator_source,
                actuator_artifact_sha256=args.movable_actuator_sha256,
                sheet_thickness=args.movable_sheet_thickness,
                sheet_thickness_method=args.movable_sheet_thickness_method,
                sheet_thickness_source=args.movable_sheet_thickness_source,
                sheet_thickness_artifact_sha256=(
                    args.movable_sheet_thickness_sha256
                ),
                sheet_thickness_geometry_fingerprint_sha256=(
                    args.movable_sheet_geometry_sha256
                ),
                sheet_thickness_fluid_name=(
                    args.movable_sheet_thickness_fluid
                ),
                sheet_thickness_opening_range=movable_sheet_opening_range,
                sheet_thickness_pressure_drop_range=movable_sheet_dp_range,
                sheet_thickness_mass_flow_range=(
                    movable_sheet_mass_flow_range
                ),
            ),
            feed_system=_feed_system_spec_from_args(args),
        )
        try:
            coupling_cooling = CoolingSpec(
                method="regenerative" if args.regen else "none",
                coolant=args.coolant,
                coolant_mass_flow=args.coolant_mdot,
                channel_count=args.channels,
                channel_width=args.channel_width,
                channel_height=args.channel_height,
                channel_roughness=args.channel_roughness,
                coolant_outlet_pressure=args.coolant_outlet_pressure,
                injector_pressure_drop=args._regen_injector_pressure_drop,
                max_wall_temperature=args.wall_temp_limit,
                **args._cooling_options,
            )

            iteration_material = (
                MaterialSpec.from_catalog(args._material.name)
                if args._material else MaterialSpec(conductivity=args.wall_k)
            )
            iteration_material.conductivity = args.wall_k

            def evaluate_cli_cooling(total_mdot, gas_propellant):
                if not args.regen:
                    return coupling_cooling, cooling_result, None
                from dataclasses import replace
                from raosim.physics import (
                    bartz_heat_flux,
                    regenerative_cooling_analysis,
                )

                fuel_mdot = total_mdot / (1.0 + args.mixture_ratio)
                trial_spec = replace(
                    coupling_cooling, coolant_mass_flow=fuel_mdot
                )
                trial_heat_flux = bartz_heat_flux(
                    contour, args.pc, gas_propellant, wall_temperature=900.0
                )
                trial_cooling = regenerative_cooling_analysis(
                    trial_heat_flux,
                    contour,
                    trial_spec,
                    iteration_material,
                    args.wall_thickness,
                    gas_propellant,
                    args.pc,
                    helix_turns=args.helix_turns,
                    wall_profile=getattr(args, "_wall_profile", None),
                    curvature_correction=args.curvature_correction,
                    coolant_outlet_pressure=args.coolant_outlet_pressure,
                    injector_pressure_drop=args._regen_injector_pressure_drop,
                )
                return trial_spec, trial_cooling, trial_heat_flux

            def evaluate_cli_injector(
                total_mdot,
                gas_propellant,
                *,
                iteration_cooling=coupling_cooling,
                iteration_cooling_result=cooling_result,
            ):
                mdot_fuel = total_mdot / (1.0 + args.mixture_ratio)
                mdot_oxidizer = args.mixture_ratio * mdot_fuel
                return evaluate_pintle_injector(
                    inj_spec,
                    mdot_fuel=mdot_fuel,
                    mdot_oxidizer=mdot_oxidizer,
                    Pc=args.pc, mixture_ratio=args.mixture_ratio,
                    chamber_radius=chamber["Rc"],
                    chamber_length=chamber["injector_to_throat_length"],
                    gamma=gas_propellant.gamma,
                    Tc=gas_propellant.Tc,
                    R_gas=gas_propellant.R_gas,
                    fuel_name=args._fuel_name,
                    oxidizer_name=args._ox_name,
                    cooling=iteration_cooling,
                    cooling_result=iteration_cooling_result,
                )

            if args.spray_cstar_coupling:
                from raosim.spray_coupling import (
                    SprayCStarCouplingSpec,
                    solve_spray_cstar_fixed_point,
                )

                coupling_spec = SprayCStarCouplingSpec(
                    enabled=True,
                    eta_mixing=args.spray_eta_mixing,
                    eta_combustion=args.spray_eta_combustion,
                    relaxation=args.spray_coupling_relaxation,
                    relative_tolerance=args.spray_coupling_tolerance,
                    max_iterations=args.spray_coupling_max_iterations,
                )

                def trial_propellant(eta_cstar):
                    trial = custom_propellant(
                        gamma=args.gamma,
                        Mw=prop.Mw,
                        Tc=prop.Tc,
                        OF=prop.OF,
                        eta_cstar=eta_cstar,
                        eta_CF=prop.eta_CF,
                        source=prop.source,
                    )
                    trial.name = prop.name
                    # Preserve a RocketCEA-provided ideal c-star rather than
                    # reconstructing it from the chamber snapshot.
                    trial.c_star = prop.c_star
                    return trial

                def spray_evaluator(eta_cstar, total_mdot):
                    trial_prop = trial_propellant(eta_cstar)
                    trial_cooling_spec, trial_cooling, trial_heat_flux = (
                        evaluate_cli_cooling(total_mdot, trial_prop)
                    )
                    trial_injector = evaluate_cli_injector(
                        total_mdot,
                        trial_prop,
                        iteration_cooling=trial_cooling_spec,
                        iteration_cooling_result=trial_cooling,
                    )
                    if trial_injector.atomization is None:
                        raise RuntimeError(
                            "injector did not return an atomization result"
                        )
                    fuel_mdot = total_mdot / (1.0 + args.mixture_ratio)
                    state = SprayRegenIterationPayload(
                        injector=trial_injector,
                        thermal=(trial_heat_flux or {}),
                        cooling=(trial_cooling or {"method": "none"}),
                        total_mass_flow=total_mdot,
                        fuel_mass_flow=fuel_mdot,
                        oxidizer_mass_flow=args.mixture_ratio * fuel_mdot,
                        coolant_mass_flow=(
                            fuel_mdot if args.regen else 0.0
                        ),
                        fuel_film_mass_flow=0.0,
                    )
                    return trial_injector.atomization, state

                spray_coupling = solve_spray_cstar_fixed_point(
                    coupling_spec,
                    initial_eta_cstar=prop.eta_cstar,
                    ideal_cstar=perf.c_star,
                    chamber_pressure=args.pc,
                    throat_area=math.pi * args.rt ** 2,
                    evaluator=spray_evaluator,
                )
                prop = trial_propellant(spray_coupling.eta_cstar)
                perf = compute_engine_performance(
                    Pc=args.pc,
                    Pa=Pa,
                    Rt=args.rt,
                    epsilon=args.epsilon,
                    prop=prop,
                    frozen_gas=args._frozen_gas,
                )
                args._prop = prop
                args._performance = perf
                args._mdot = perf.m_dot
                args._mdot_f = perf.m_dot / (1.0 + args.mixture_ratio)
                args._mdot_o = args.mixture_ratio * args._mdot_f
                final_state = spray_coupling.payload
                if not isinstance(final_state, SprayRegenIterationPayload):
                    raise RuntimeError(
                        "spray/c-star evaluator returned an unexpected final state"
                    )
                inj = final_state.injector
                if args.regen:
                    cooling_result = final_state.cooling
                    args.coolant_mdot = final_state.coolant_mass_flow
                    summary["cooling"] = _cooling_summary_payload(
                        cooling_result, args
                    )
                    from dataclasses import replace
                    final_cooling_spec = replace(
                        coupling_cooling,
                        coolant_mass_flow=final_state.coolant_mass_flow,
                    )
                    analysis_wall_profile = getattr(args, "_wall_profile", None)
                    final_wall_thickness = (
                        analysis_wall_profile.t_hot
                        if analysis_wall_profile is not None
                        else args.wall_thickness
                    )
                    if iteration_material.elastic_modulus:
                        from raosim.physics import (
                            channel_pressure_hoop_radius,
                            coaxial_shell_wall_stress_profile,
                        )
                        final_stress = coaxial_shell_wall_stress_profile(
                            pressure_differential=cooling_result[
                                "liner_pressure_differential"
                            ],
                            inner_radius=channel_pressure_hoop_radius(
                                args.channel_width, final_wall_thickness
                            ),
                            wall_thickness=final_wall_thickness,
                            heat_flux=cooling_result["q"],
                            elastic_modulus=iteration_material.elastic_modulus,
                            thermal_expansion=iteration_material.thermal_expansion,
                            poisson_ratio=iteration_material.poisson_ratio,
                            conductivity=args.wall_k,
                            yield_strength=iteration_material.yield_strength,
                        )
                        summary["structural"] = {
                            "combined_stress_MPa": (
                                final_stress["combined_stress"] / 1e6
                            ),
                            "thermal_stress_MPa": (
                                final_stress["thermal_stress"] / 1e6
                            ),
                            "pressure_stress_MPa": (
                                final_stress["pressure_stress"] / 1e6
                            ),
                            "yield_strength_MPa": (
                                iteration_material.yield_strength / 1e6
                            ),
                            "stress_margin": final_stress["stress_margin"],
                            "model": final_stress["model"],
                            "governing_index": final_stress["governing_index"],
                            "max_liner_pressure_differential_bar": float(
                                max(cooling_result["liner_pressure_differential"])
                                / 1e5
                            ),
                            "outer_loop_state": (
                                "final_spray_cstar_regen_iterate"
                            ),
                        }
                    # Previously generated thermal/structural figures used the
                    # pre-coupling flow.  Overwrite them from the final iterate.
                    try:
                        from raosim.plotting import (
                            plot_cooling_profile,
                            plot_coolant_channel_march,
                        )
                        figure = plot_cooling_profile(
                            cooling_result,
                            contour=contour,
                            max_wall_temperature=getattr(
                                iteration_material, "max_temperature", None
                            ),
                            save_path=args.out / "cooling_profile.png",
                            show=show,
                        )
                        figure.clf()
                        figure = plot_coolant_channel_march(
                            cooling_result,
                            save_path=args.out / "channel_march.png",
                            show=show,
                        )
                        figure.clf()
                        for name in ("cooling_profile.png", "channel_march.png"):
                            if name not in artifacts:
                                artifacts.append(name)
                    except Exception as exc:
                        print(yellow(
                            f"    final coupled cooling plots skipped: {exc}"
                        ))
                    try:
                        from raosim.physics import wall_cross_section_field
                        from raosim.plotting import plot_channel_cross_section
                        final_cross_section = wall_cross_section_field(
                            cooling_result,
                            final_state.thermal,
                            contour,
                            final_cooling_spec,
                            iteration_material,
                            final_wall_thickness,
                            prop,
                            args.pc,
                            station="peak",
                        )
                        figure = plot_channel_cross_section(
                            final_cross_section,
                            save_path=args.out / "cross_section.png",
                            show=show,
                        )
                        figure.clf()
                        if "cross_section.png" not in artifacts:
                            artifacts.append("cross_section.png")
                    except Exception as exc:
                        print(yellow(
                            f"    final coupled cross-section skipped: {exc}"
                        ))
                    if iteration_material.elastic_modulus:
                        try:
                            from raosim.plotting import (
                                plot_structural_life_dashboard,
                            )
                            figure = plot_structural_life_dashboard(
                                cooling_result,
                                final_stress,
                                material=iteration_material,
                                required_cycles=getattr(
                                    args, "required_cycles", None
                                ),
                                save_path=args.out / "structural_life.png",
                                show=show,
                            )
                            figure.clf()
                            if "structural_life.png" not in artifacts:
                                artifacts.append("structural_life.png")
                        except Exception as exc:
                            print(yellow(
                                "    final coupled structural plot skipped: "
                                f"{exc}"
                            ))
                summary["spray_cstar_coupling"] = {
                    "enabled": True,
                    **spray_coupling.to_dict(),
                }
                summary["performance"].update({
                    "c_star_effective_m_s": perf.c_star_effective,
                    "eta_cstar": perf.eta_cstar,
                    "eta_Isp": perf.eta_Isp,
                    "Isp_s": perf.Isp,
                    "mdot_total_kg_s": perf.m_dot,
                    "mdot_fuel_kg_s": args._mdot_f,
                    "mdot_oxidizer_kg_s": args._mdot_o,
                })
                summary["thrust_closure"].update({
                    "spray_cstar_fixed_point_enabled": True,
                    "effective_cstar_m_s": perf.c_star_effective,
                    "required_mass_flow_kg_s": perf.m_dot,
                })
                print(dim(
                    "    spray/c-star fixed point: "
                    f"eta_cstar={perf.eta_cstar:.4f}, "
                    f"mdot={perf.m_dot:.4f} kg/s, "
                    f"{len(spray_coupling.iterations)} iterations"
                ))
            else:
                inj = evaluate_cli_injector(perf.m_dot, prop)
                summary["spray_cstar_coupling"] = {
                    "enabled": False,
                    "status": "disabled",
                    "reason": "vaporization screen is report-only by default",
                }
        except (InjectorUnsupportedState, InjectorSpecError, RuntimeError, ValueError) as exc:
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
        if args.electric_pump:
            print("\n" + cyan("▸ " + bold("Electric pump")) +
                  dim("  (drive, battery, impeller, inducer, diffuser)"))
            from raosim.pumps import size_electric_pumps

            pump = size_electric_pumps(
                inj.feed_system,
                _pump_spec_from_args(args),
            )
            _print_pump_panel(pump)
            pump_dict = pump.to_dict()
            summary["electric_pump"] = pump_dict
            (args.out / "pump.json").write_text(json.dumps(pump_dict, indent=2))
            artifacts.append("pump.json")
            print(green("    wrote pump.json"))
            if pump_dict.get("hardware_bom"):
                (args.out / "pump_bom.json").write_text(
                    json.dumps(pump_dict["hardware_bom"], indent=2)
                )
                artifacts.append("pump_bom.json")
                print(green("    wrote pump_bom.json"))
            reference_geometry = {
                role: line["reference_geometry"]
                for role, line in pump_dict.get("lines", {}).items()
                if line.get("reference_geometry") is not None
            }
            if reference_geometry:
                (args.out / "pump_reference_geometry.json").write_text(
                    json.dumps(reference_geometry, indent=2)
                )
                artifacts.append("pump_reference_geometry.json")
                print(green("    wrote pump_reference_geometry.json"))
            if args.pump_cad != "none":
                try:
                    from raosim.pump_cad import export_pump_package

                    pump_cad_mode = (
                        "parts" if args.pump_cad == "auto" else args.pump_cad
                    )
                    fmt = args.pump_cad_format
                    # The mesh writer handles STL; parameters json/csv are
                    # always written.  step/both add the CadQuery B-rep
                    # package (true parts + named assemblies).
                    pkg = export_pump_package(
                        pump,
                        args.out / "pump",
                        cad=pump_cad_mode if fmt != "step" else "none",
                        cad_format="stl",
                        allow_open_mesh=args.allow_open_pump_mesh,
                    )
                    summary["pump_package"] = {
                        "dir": pkg["dir"],
                        "files": pkg["files"],
                        "notes": pkg["notes"],
                    }
                    if fmt in ("step", "both"):
                        from raosim.pump_cad_brep import (
                            export_pump_brep_package,
                        )

                        brep = export_pump_brep_package(
                            pump, args.out / "pump"
                        )
                        pkg["files"].update(brep["files"])
                        pkg["notes"].extend(brep["notes"])
                        summary["pump_package"]["step_representation"] = (
                            brep["step_representation"]
                        )
                        summary["pump_package"]["brep_diagnostics"] = {
                            key: {
                                "valid": info["valid"],
                                "solid_count": info["solid_count"],
                                "volume_mm3": info["volume_mm3"],
                            }
                            for key, info in brep["diagnostics"].items()
                        }
                        summary["pump_package"]["assembly_gates"] = brep[
                            "assembly_gates"
                        ]
                        summary["pump_package"]["cold_flow_release_ready"] = (
                            brep["cold_flow_release_ready"]
                        )
                        summary["pump_package"]["hardware_qualified"] = brep[
                            "hardware_qualified"
                        ]
                        summary["pump_package"]["external_release_blockers"] = (
                            brep["external_release_blockers"]
                        )
                        failed_reference_gates = [
                            name
                            for name, gate_info in brep["assembly_gates"].items()
                            if not bool(gate_info.get("passed", False))
                        ]
                        if failed_reference_gates:
                            print(yellow(
                                "    pump B-rep remains packaging-reference "
                                "geometry; unresolved gates: "
                                + ", ".join(failed_reference_gates)
                            ))
                    for path in pkg["files"].values():
                        artifacts.append(os.path.relpath(str(path), str(args.out)))
                    print(green(
                        f"    wrote pump/ package ({len(pkg['files'])} files)"
                    ))
                    for note in pkg["notes"]:
                        print(dim(f"    pump package: {note}"))
                except Exception as exc:
                    message = (
                        "requested pump CAD package failed a required "
                        f"geometry/export gate: {exc}"
                    )
                    print(red(f"    {message}"))
                    summary["pump_package"] = {
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    (args.out / "summary.json").write_text(
                        json.dumps(summary, indent=2)
                    )
                    return 2
            if args.pump_visualize:
                try:
                    from raosim.flow_viz import animate_pump_particles

                    viz_role = (
                        "fuel"
                        if pump.lines.get("fuel")
                        and pump.lines["fuel"].impeller is not None
                        else next(
                            role for role, line in pump.lines.items()
                            if line.impeller is not None
                        )
                    )
                    animate_pump_particles(
                        pump,
                        role=viz_role,
                        save_path=args.out / "pump_particles.gif",
                        fps=25,
                        show=show,
                    )
                    artifacts.append("pump_particles.gif")
                    print(green("    wrote pump_particles.gif")
                          + (dim("  (window)") if show else ""))
                except Exception as exc:
                    print(yellow(f"    pump visualization skipped: {exc}"))
            if not pump.feasible and not args.allow_infeasible_pump:
                print(red("    electric pump gates FAILED — blocking chamber "
                          "export; re-run with --allow-infeasible-pump to "
                          "override."))
                (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
                return 2
            print(green("    electric pump sized ✓") if pump.feasible else
                  yellow("    electric pump sized with FAILING gates "
                         "(--allow-infeasible-pump)"))
        if not inj.feasible and not args.allow_infeasible_injector:
            print(red("    injector gates FAILED — blocking chamber export; "
                      "re-run with --allow-infeasible-injector to override."))
            (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
            return 2
        print(green("    injector sized ✓") if inj.feasible else
              yellow("    injector sized with FAILING gates "
                     "(--allow-infeasible-injector)"))
        try:
            from raosim.injector_cad import resolve_machined_pintle_layout
            from raosim.interface import resolve_bolted_interface_geometry

            layout = resolve_machined_pintle_layout(inj, spec=inj_spec)
            lr = layout["resolved"]
            interface_resolution = resolve_bolted_interface_geometry(
                chamber_pressure=args.pc,
                chamber_radius=chamber["Rc"],
                wall_thickness=args.wall_thickness,
                flange_outer_diameter=args.flange_od,
                flange_length=args.flange_length,
                face_outer_diameter=args.injector_face_od,
                face_thickness=args.injector_face_thickness,
                bolt_count=args.bolt_count,
                bolt_circle_diameter=args.bolt_circle,
                bolt_hole_diameter=args.bolt_hole,
                bolt_diameter=args.bolt_diameter,
                bolt_allowable_stress=args.bolt_allowable_stress,
                material_yield_strength=(mat.yield_strength if mat else None),
                min_feature=args.injector_min_feature,
                min_tool_diameter=args.min_tool_diameter,
                minimum_face_outer_diameter=lr["faceplate_outer_diameter_m"],
                minimum_face_thickness=lr["faceplate_thickness_m"],
                minimum_bolt_circle_diameter=lr["bolt_circle_diameter_m"],
                minimum_bolt_hole_diameter=lr["bolt_hole_diameter_m"],
                joint_separation_factor=args.joint_separation_factor,
            )
            args.flange_od = interface_resolution.flange_outer_diameter
            args.flange_length = interface_resolution.flange_length
            args.injector_face_od = interface_resolution.face_outer_diameter
            args.injector_face_thickness = interface_resolution.face_thickness
            args.bolt_count = interface_resolution.bolt_count
            args.bolt_circle = interface_resolution.bolt_circle_diameter
            args.bolt_hole = interface_resolution.bolt_hole_diameter
            args._interface_seal_center = (
                lr["seal_center_radius_m"]
                if lr["seal_type"] == "o_ring" else None
            )
            args._interface_seal_width = (
                lr["o_ring_groove_width_m"]
                if lr["seal_type"] == "o_ring" else None
            )
            inj_spec.geometry.face_od = args.injector_face_od
            inj_spec.geometry.face_thickness = args.injector_face_thickness
            inj_spec.mechanical.bolt_count = args.bolt_count
            inj_spec.mechanical.bolt_circle_diameter = args.bolt_circle
            inj_spec.mechanical.bolt_hole_diameter = args.bolt_hole
            inj_spec.mechanical.faceplate_outer_diameter = args.injector_face_od
            inj_spec.mechanical.faceplate_thickness = args.injector_face_thickness
            summary["injector_interface_resolution"] = (
                interface_resolution.to_dict()
            )
            auto = interface_resolution.auto_sized_fields
            print(dim(
                "    final injector/chamber interface: "
                f"flange OD {args.flange_od*1e3:.1f} mm, "
                f"face OD {args.injector_face_od*1e3:.1f} mm, "
                f"BCD {args.bolt_circle*1e3:.1f} mm, "
                f"{args.bolt_count}x Ø{args.bolt_hole*1e3:.1f} mm"
                + ("  (auto-sized)" if auto else "")
            ))
        except Exception as exc:
            print(yellow(f"    injector interface/layout sync skipped: {exc}"))

        # ---- optional architecture-dispatched throttle map (computed before
        #      the figures so it can be plotted alongside them) -----------
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
                    chamber_radius=chamber["Rc"],
                    chamber_length=chamber["injector_to_throat_length"],
                    gamma=prop.gamma, Tc=prop.Tc, R_gas=prop.R_gas,
                    levels=levels, pc_exponent=args.throttle_pc_exponent)
                summary["injector_throttle_map"] = tm.to_dict()
                if tm.kinematic_model is not None:
                    controller_role = tm.points[0].axial_controller_role
                    print(
                        "\n    " + bold("Throttle map")
                        + dim(
                            f"  (Pc∝f^{args.throttle_pc_exponent:g}; fixed "
                            f"hardware, physical radial travel + separate "
                            f"{controller_role} controller)"
                        )
                    )
                    print(
                        f"      {'f':>5} {'Pc[bar]':>8} {'Lopen[mm]':>9} "
                        f"{'stroke':>7} {'χ_ax':>7} {'TMR':>6} "
                        f"{'SMD[µm]':>8} {'feas':>5}"
                    )
                    for p in tm.points:
                        badge = green("yes") if p.feasible else red("no")
                        print(
                            f"      {p.throttle:>5.2f} {p.Pc/1e5:>8.1f} "
                            f"{p.radial_opening*1e3:>9.4f} "
                            f"{p.actuator_stroke_fraction:>7.3f} "
                            f"{p.required_axial_controller_dp_fraction:>7.4f} "
                            f"{p.total_momentum_ratio:>6.3f} "
                            f"{p.smd_limiting*1e6:>8.0f} {badge:>5}"
                        )
                else:
                    print(
                        "\n    " + bold("Throttle map")
                        + dim(
                            f"  (Pc∝f^{args.throttle_pc_exponent:g}; "
                            "commanded hydraulic areas, no actuator "
                            "kinematics)"
                        )
                    )
                    print(
                        f"      {'f':>5} {'Pc[bar]':>8} {'A_a/Afull':>9} "
                        f"{'v_a':>6} {'TMR':>6} {'SMD[µm]':>8} "
                        f"{'η_vap':>6} {'feas':>5}"
                    )
                    for p in tm.points:
                        badge = green("yes") if p.feasible else red("no")
                        print(
                            f"      {p.throttle:>5.2f} {p.Pc/1e5:>8.1f} "
                            f"{p.annulus_area_command_fraction:>9.3f} "
                            f"{p.v_annulus:>6.0f} "
                            f"{p.total_momentum_ratio:>6.3f} "
                            f"{p.smd_limiting*1e6:>8.0f} "
                            f"{p.eta_vaporization:>6.2f} {badge:>5}"
                        )
            except Exception as exc:
                message = f"requested throttle map is not reachable: {exc}"
                print(red(f"    {message}"))
                summary["injector_throttle_map"] = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                (args.out / "summary.json").write_text(
                    json.dumps(summary, indent=2)
                )
                return 2

        # ---- pintle deliverable package (mandatory schematic + table) ---
        try:
            from raosim.injector_export import export_pintle_package

            cad_mode = args.injector_cad
            cad_format = args.injector_cad_format
            if args.injector_cad == "step":
                cad_mode = "machined"
                cad_format = "step"
            elif args.injector_cad == "auto":
                cad_mode = "machined"
                cad_format = "step"

            pkg = export_pintle_package(
                inj, args.out / "pintle", spec=inj_spec,
                cad=cad_mode, cad_format=cad_format,
                movable_sleeve=args.pintle_sleeve,
                radial_style=args.pintle_radial_exit)
            summary["injector_package"] = {
                "dir": pkg["dir"],
                "files": pkg["files"],
                "cad_audit": pkg.get("cad_audit"),
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
            if args.injector_cad != "none":
                message = (
                    "requested pintle CAD package failed a required "
                    f"geometry/export gate: {exc}"
                )
                print(red(f"    {message}"))
                summary["injector_package"] = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                (args.out / "summary.json").write_text(
                    json.dumps(summary, indent=2)
                )
                return 2
            print(yellow(f"    pintle diagnostic package skipped: {exc}"))

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
        from raosim.interface import (
            screen_composite_regen_wall,
            screen_injector_chamber_interface,
        )

        composite_wall_screen = None
        sized_interface_profile = getattr(args, "_wall_profile", None)
        if cooling_result is not None and sized_interface_profile is not None and mat:
            from raosim.design import MaterialSpec

            liner_screen_material = MaterialSpec.from_catalog(args._material.name)
            liner_screen_material.conductivity = float(args.wall_k)
            liner_screen_material.max_temperature = float(args.wall_temp_limit)
            jacket_screen_material = (
                MaterialSpec.from_catalog(args.jacket_material)
                if args.jacket_material else liner_screen_material
            )
            composite_wall_screen = screen_composite_regen_wall(
                chamber_pressure=args.pc,
                wall_profile=sized_interface_profile,
                liner_material=liner_screen_material,
                jacket_material=jacket_screen_material,
                structural_fos=args.structural_fos,
                gas_side_wall_temperature=
                    cooling_result["gas_side_wall_temperature"],
                coolant_side_wall_temperature=
                    cooling_result["coolant_side_wall_temperature"],
                coolant_temperature=cooling_result["coolant_temperature"],
                coolant_pressure=cooling_result["coolant_pressure"],
                liner_pressure_differential=
                    cooling_result["liner_pressure_differential"],
                heat_flux=cooling_result["q"],
                screen_station_index=0,
                screen_selection="injector_face_chamber_station",
            )

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
            composite_wall_screen=composite_wall_screen,
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
                bolt_count=args.bolt_count,
                bolt_circle_diameter=args.bolt_circle,
                bolt_hole_diameter=args.bolt_hole,
                seal_center_radius=getattr(
                    args, "_interface_seal_center", None
                ),
                seal_groove_width=getattr(
                    args, "_interface_seal_width", None
                ),
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
                    release_mode=args.regen_release_mode,
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
                    "release_mode": rb["release_mode"],
                    "flow_path_status": rb["flow_path_status"],
                    "cold_flow_geometry_ready": rb[
                        "cold_flow_geometry_ready"
                    ],
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

    # ---- optional engine-level assembly (pump CAD plan Phase 3) ----------
    if getattr(args, "engine_assembly", False):
        try:
            from raosim.engine_cad import (
                cadquery_available as _engine_cq_ok,
                export_engine_assembly,
            )

            if not _engine_cq_ok():
                raise RuntimeError(
                    "--engine-assembly requires CadQuery/OpenCascade "
                    "(pip install cadquery)"
                )
            candidates = {
                "wall": args.out / "wall.step",
                "jacket": args.out / "jacket.step",
                "pintle_injector": next(
                    (p for p in (
                        args.out / "pintle" / "injector_assembly_machined.step",
                        args.out / "pintle" / "pintle_assembly.step",
                        args.out / "pintle" / "pintle_reference.step",
                    ) if p.exists()),
                    args.out / "pintle" / "pintle_reference.step",
                ),
                "fuel_pump": args.out / "pump" / "pump_brep" / "fuel_pump.step",
                "oxidizer_pump": (
                    args.out / "pump" / "pump_brep" / "oxidizer_pump.step"
                ),
                "shared_battery_pack": (
                    args.out / "pump" / "pump_brep" / "shared_battery_pack.step"
                ),
            }
            engine_info = export_engine_assembly(
                args.out / "engine_assembly.step",
                {k: v for k, v in candidates.items() if v.exists()},
                pump_result=summary.get("electric_pump"),
            )
            summary["engine_assembly"] = engine_info
            artifacts.append("engine_assembly.step")
            print(green(
                f"    wrote engine_assembly.step "
                f"({len(engine_info['children'])} placed bodies)"
            ))
            for note in engine_info["notes"]:
                print(dim(f"    engine assembly: {note}"))
        except Exception as exc:
            raise RuntimeError(
                "requested engine assembly failed a required placement/export "
                f"gate: {exc}"
            ) from exc

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    artifacts.append("summary.json")

    # ---- results panel ----------------------------------------------------
    print("\n" + green("▸ " + bold("Done")) + f"  →  {bold(str(args.out))}/")
    for a in artifacts:
        print(f"    {dim('•')} {a}")
    release_report = args._release_readiness
    if release_report.blocked:
        print(yellow(
            "\n  Physical release BLOCKED: "
            f"{len(release_report.blockers)} external evidence requirements "
            "are missing, invalid, or failed (see summary.json)."
        ))
    else:
        print(dim(
            "\n  Release evidence is complete for this manifest; hardware "
            "qualification still requires the external engineering authority."
        ))
    print(dim("  Numerical design/CAD output is never hardware-qualified by LREKit."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
