#!/usr/bin/env python3
"""
main.py – CLI for the Rao Bell Nozzle Design Toolbox.

Usage:
    RaoRocketSim                          # interactive mode
    RaoRocketSim --help                   # show all flags
    RaoRocketSim --Rt 25 --Pc 60 \\
        --propellant LOX/LCH4 --epsilon 12 \\
        --output nozzle.csv               # batch mode (no prompts)
    RaoRocketSim --sweep epsilon 4 50 20  # parameter sweep
"""

from __future__ import annotations
import argparse
import json
import math
import sys
import numpy as np
from pathlib import Path

from raosim.cea import propellant_from_request
from raosim.design import (
    CoolingSpec,
    DesignInput,
    InterfaceSpec,
    ManufacturingSpec,
    MaterialSpec,
    MissionAmbientSpec,
    ThermoSpec,
    design_nozzle_v2,
    throat_radius_for_target_thrust,
)
from raosim.propellants import (
    get_propellant, custom_propellant, list_propellants, Propellant,
)
from raosim.gas_dynamics import (
    mach_from_area_ratio, expansion_ratio_from_pressure,
    isentropic_pressure_ratio, prandtl_meyer,
)
from raosim.nozzle_geometry import bell_nozzle_contour, lookup_angles
from raosim.engine import compute_engine_performance, g0
from raosim.injector import (
    FeedLineSpec,
    FeedSystemSpec,
    InjectorManufacturingSpec,
    InjectorSpec,
    PintleMechanicalSpec,
    PintleGeometrySpec,
    PropellantFeedSpec,
)
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
  Interactive:   RaoRocketSim
  Batch:         RaoRocketSim --propellant LOX/RP-1 --Pc 45 --Rt 20 --epsilon 10
  Sweep:         RaoRocketSim --propellant LOX/LCH4 --Pc 60 --Rt 25 \\
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
    p.add_argument('--mode', type=str, default='preliminary',
                   choices=['preliminary', 'validated'],
                   help='Design workflow mode for the v2 schema')
    p.add_argument('--thermo', type=str, default='constant_gamma',
                   choices=['constant_gamma', 'cea_frozen', 'cea_equilibrium'],
                   help='Thermochemistry model for the v2 schema')
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
                    choices=['bezier', 'moc', 'rao', 'rao_variational_moc'],
                    help='Contour method: bezier (default), moc (direct wall optimization), '
                         'rao (legacy experimental variational path), or '
                         'rao_variational_moc (auditable residual/MOC path)')
    p.add_argument('--compare', action='store_true',
                    help='Compare conical, Bézier, and Rao contours side by side')
    p.add_argument('--benchmark-case', type=str, default=None,
                   help='Run a literature benchmark case by id')
    p.add_argument('--benchmark-method', type=str, default='bezier',
                   choices=['bezier', 'moc', 'rao', 'all'],
                   help='Benchmark solver method (default bezier)')
    p.add_argument('--benchmark-report', type=str, default=None,
                   help='Benchmark report file or directory')
    p.add_argument('--benchmark-strict', action='store_true',
                   help='Exit nonzero for diagnostic xfail or strict benchmark failure')
    p.add_argument('--theta-n', type=float, default=None,
                   help='Override initial wall angle θ_n [°]')
    p.add_argument('--theta-e', type=float, default=None,
                   help='Override exit wall angle θ_e [°]')
    p.add_argument('--pa-over-p0', type=float, default=0.0,
                   help='Rao variational design ambient/stagnation pressure ratio Pa/P0')
    p.add_argument('--rao-moc-n-control', type=int, default=12,
                   help='Rao variational/MOC control-surface station count')
    p.add_argument('--rao-moc-n-kernel', type=int, default=12,
                   help='Rao variational/MOC kernel/MOC point count')
    p.add_argument('--rao-moc-max-nfev', type=int, default=25,
                   help='Rao variational/MOC least-squares function-evaluation limit')
    p.add_argument('--rao-moc-skip-moc', action='store_true',
                   help='Skip raw MOC closure diagnostics for a fast residual-only Rao run')


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
                   help='Solid CAD export: STEP, legacy IPT manifest, or both')
    p.add_argument('--wall-thickness', type=float, default=None,
                   help='Solid nozzle wall thickness [mm] for STEP/STL CAD')
    p.add_argument('--flange-od', type=float, default=None,
                   help='Optional inlet flange outer diameter [mm]')
    p.add_argument('--flange-length', type=float, default=None,
                   help='Optional inlet flange axial length [mm]')
    p.add_argument('--cooling', type=str, default='none',
                   choices=['none', 'regenerative'],
                   help='Cooling screen model for v2 physics gates')
    p.add_argument('--coolant', type=str, default=None,
                   help='Coolant name for regenerative screening')
    p.add_argument('--channel-count', type=int, default=None,
                   help='Number of regenerative cooling channels')
    p.add_argument('--channel-width', type=float, default=None,
                   help='Cooling channel width [mm]')
    p.add_argument('--channel-height', type=float, default=None,
                   help='Cooling channel height [mm]')
    p.add_argument('--coolant-mdot', type=float, default=None,
                   help='Coolant mass flow [kg/s]')
    p.add_argument('--coolant-cp', type=float, default=3500.0,
                   help='Coolant heat capacity [J/kg/K]')
    p.add_argument('--coolant-inlet-temp', type=float, default=None,
                   help='Coolant inlet temperature [K] (fluid default if omitted)')
    p.add_argument('--max-wall-temp', type=float, default=950.0,
                   help='Cooling wall temperature limit [K]')
    p.add_argument('--material', type=str, default='Inconel 718',
                   help='Material name for v2 screening')
    p.add_argument('--yield-strength', type=float, default=900.0,
                   help='Material yield strength [MPa]')
    p.add_argument('--material-k', type=float, default=16.0,
                   help='Material thermal conductivity [W/m/K]')
    p.add_argument('--material-max-temp', type=float, default=1250.0,
                   help='Material max screening temperature [K]')
    p.add_argument('--max-heat-flux', type=float, default=25.0,
                   help='Material heat flux screening limit [MW/m^2]')
    p.add_argument('--bolt-count', type=int, default=None,
                   help='Optional flange bolt count')
    p.add_argument('--bolt-circle', type=float, default=None,
                   help='Optional bolt circle diameter [mm]')
    p.add_argument('--bolt-hole', type=float, default=None,
                   help='Optional bolt hole diameter [mm]')
    p.add_argument('--bolt-diameter', type=float, default=None,
                   help='Optional actual bolt/tensile diameter [mm]')
    p.add_argument('--bolt-allowable-stress', type=float, default=None,
                   help='Optional bolt allowable tensile stress [MPa]')
    p.add_argument('--joint-separation-factor', type=float, default=1.5,
                   help='Clamp-load factor on Pc*pi*Rc^2 separating load')
    p.add_argument('--injector-face-od', type=float, default=None,
                   help='Optional injector face outer diameter [mm]')
    p.add_argument('--injector-face-thickness', type=float, default=None,
                   help='Optional injector faceplate thickness [mm]')
    p.add_argument('--injector', choices=['none', 'pintle'], default='none',
                   help='Integrated injector model for the v2 workflow')
    p.add_argument('--fuel-injector-dp-fraction', type=float, default=0.2,
                   help='Fuel injector pressure drop divided by Pc')
    p.add_argument('--oxidizer-injector-dp-fraction', type=float, default=0.2,
                   help='Oxidizer injector pressure drop divided by Pc')
    p.add_argument('--fuel-inlet-temperature', type=float, default=None,
                   help='Fuel injector inlet temperature [K]')
    p.add_argument('--oxidizer-inlet-temperature', type=float, default=None,
                   help='Oxidizer injector inlet temperature [K]')
    p.add_argument('--fuel-inlet-pressure', type=float, default=None,
                   help='Fuel injector inlet pressure [bar]')
    p.add_argument('--oxidizer-inlet-pressure', type=float, default=None,
                   help='Oxidizer injector inlet pressure [bar]')
    p.add_argument('--pintle-radial-stream', choices=['fuel', 'oxidizer'],
                   default='fuel', help='Pintle radial/slotted stream')
    p.add_argument('--pintle-diameter', type=float, default=None,
                   help='Pintle diameter [mm]; defaults from chamber diameter')
    p.add_argument('--pintle-slot-count', type=int, default=24,
                   help='Number of radial pintle slots')
    p.add_argument('--pintle-slot-aspect-ratio', type=float, default=1.0,
                   help='Auto-sized slot height/width')
    p.add_argument('--pintle-deflector-angle', type=float, default=0.0,
                   help='Pintle deflector angle [deg]')
    p.add_argument('--pintle-target-momentum-ratio', type=float, default=None,
                   help='Optional target radial/axial momentum ratio')
    p.add_argument('--injector-min-feature', type=float, default=0.3,
                   help='Minimum injector slot/gap/web feature [mm]')
    p.add_argument('--allow-infeasible-injector', action='store_true',
                   help='Return preliminary output despite failed injector gates')
    # --- feed-system / pump closure (feed-pressure ledger) --------------
    p.add_argument('--feed-architecture', choices=['pump_fed', 'pressure_fed'],
                   default='pump_fed',
                   help='Feed-system architecture label for the pump/tank closure')
    p.add_argument('--fuel-supply-pressure', type=float, default=None,
                   help='Available fuel pump/tank outlet pressure [bar]')
    p.add_argument('--oxidizer-supply-pressure', type=float, default=None,
                   help='Available oxidizer pump/tank outlet pressure [bar]')
    p.add_argument('--fuel-flow-capacity', type=float, default=None,
                   help='Available fuel pump mass-flow capacity [kg/s]')
    p.add_argument('--oxidizer-flow-capacity', type=float, default=None,
                   help='Available oxidizer pump mass-flow capacity [kg/s]')
    p.add_argument('--fuel-line-loss-fraction', type=float, default=0.0,
                   help='Fuel line+valve loss as a fraction of Pc (default 0)')
    p.add_argument('--oxidizer-line-loss-fraction', type=float, default=0.0,
                   help='Oxidizer line+valve loss as a fraction of Pc (default 0)')
    p.add_argument('--fuel-manifold-loss-fraction', type=float, default=0.0,
                   help='Fuel manifold-loss ALLOWANCE charged to the budget '
                        '(fraction of Pc; screen value reported separately)')
    p.add_argument('--oxidizer-manifold-loss-fraction', type=float, default=0.0,
                   help='Oxidizer manifold-loss allowance (fraction of Pc)')
    p.add_argument('--feed-control-margin-fraction', type=float, default=0.0,
                   help='Control/transient margin held above the steady '
                        'requirement on both lines (fraction of Pc)')
    p.add_argument('--fuel-tank-pressure', type=float, default=None,
                   help='Fuel tank/inlet pressure for pump head + NPSH [bar]')
    p.add_argument('--oxidizer-tank-pressure', type=float, default=None,
                   help='Oxidizer tank/inlet pressure for pump head + NPSH [bar]')
    p.add_argument('--fuel-npsh-required', type=float, default=None,
                   help='Fuel pump required NPSH [bar]')
    p.add_argument('--oxidizer-npsh-required', type=float, default=None,
                   help='Oxidizer pump required NPSH [bar]')
    # --- injector reference-geometry / CAD output ----------------------
    p.add_argument('--injector-cad',
                   choices=['none', 'reference', 'parts', 'machined'],
                   default='none',
                   help='Pintle CAD output: none (schematic+JSON+CSV only, '
                        'always written), reference (single file), parts '
                        '(reference + named part files), or machined '
                        '(Boolean-cut STEP bodies + report)')
    p.add_argument('--injector-cad-format', choices=['step', 'stl', 'dxf'],
                   default='step',
                   help='Pintle CAD format: step (portable B-rep, default), '
                        'stl (mesh preview), or dxf (2-D meridional profile)')
    p.add_argument('--interface-length', type=float, default=None,
                   help='Optional chamber/nozzle interface length [mm]')
    p.add_argument('--throat-insert', action='store_true',
                   help='Record a throat-insert CAD placeholder in v2 metadata')
    p.add_argument('--throat-insert-material', type=str, default=None,
                   help='Optional throat insert material name')
    p.add_argument('--tolerance', type=float, default=None,
                   help='Manufacturing tolerance [mm]')
    p.add_argument('--weld-allowance', type=float, default=None,
                   help='Weld allowance [mm]')
    p.add_argument('--braze-allowance', type=float, default=None,
                   help='Braze allowance [mm]')
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


def _should_use_v2(args) -> bool:
    return any([
        args.mode != 'preliminary',
        args.thermo != 'constant_gamma',
        args.cooling != 'none',
        args.channel_count is not None,
        args.channel_width is not None,
        args.channel_height is not None,
        args.coolant_mdot is not None,
        args.bolt_count is not None,
        args.bolt_circle is not None,
        args.bolt_hole is not None,
        args.bolt_diameter is not None,
        args.bolt_allowable_stress is not None,
        args.joint_separation_factor != 1.5,
        args.injector_face_od is not None,
        args.injector_face_thickness is not None,
        args.injector != 'none',
        args.interface_length is not None,
        args.throat_insert,
        args.throat_insert_material is not None,
        args.tolerance is not None,
        args.weld_allowance is not None,
        args.braze_allowance is not None,
        args.material != 'Inconel 718',
        args.yield_strength != 900.0,
        args.material_k != 16.0,
        args.material_max_temp != 1250.0,
        args.max_heat_flux != 25.0,
    ])


def _mm_to_m(value):
    return value / 1000.0 if value is not None else None


def run_batch_v2(args):
    """Non-interactive v2 design workflow."""
    _header()
    Pc = args.Pc * 1e5
    Pa = (args.Pa if args.Pa is not None else 101.325) * 1e3
    build_dir, version = create_build_dir()
    cad = args.cad
    if cad == 'none' and args.stl:
        cad = 'stl'

    design_input = DesignInput(
        thermo=ThermoSpec(
            mode=args.thermo,
            propellant_name=args.propellant,
            oxidizer=args.oxidizer,
            fuel=args.fuel,
            mixture_ratio=args.of,
            eta_Isp=args.eta,
        ),
        Pc=Pc,
        Rt=args.Rt / 1000.0 if args.Rt is not None else None,
        target_thrust=args.target_thrust,
        epsilon=args.epsilon,
        method=args.method,
        mode=args.mode,
        length_pct=args.length_pct,
        theta_n=args.theta_n,
        theta_e=args.theta_e,
        contraction_ratio=args.contraction_ratio,
        L_star=args.L_star,
        ambient=MissionAmbientSpec(Pa=Pa),
        cooling=CoolingSpec(
            method=args.cooling,
            coolant=args.coolant,
            channel_count=args.channel_count,
            channel_width=_mm_to_m(args.channel_width),
            channel_height=_mm_to_m(args.channel_height),
            coolant_mass_flow=args.coolant_mdot,
            coolant_cp=args.coolant_cp,
            coolant_inlet_temperature=args.coolant_inlet_temp,
            max_wall_temperature=args.max_wall_temp,
        ),
        material=MaterialSpec(
            name=args.material,
            yield_strength=args.yield_strength * 1e6,
            conductivity=args.material_k,
            max_temperature=args.material_max_temp,
            max_heat_flux=args.max_heat_flux * 1e6,
        ),
        interface=InterfaceSpec(
            flange_od=_mm_to_m(args.flange_od),
            flange_length=_mm_to_m(args.flange_length),
            bolt_count=args.bolt_count,
            bolt_circle_diameter=_mm_to_m(args.bolt_circle),
            bolt_hole_diameter=_mm_to_m(args.bolt_hole),
            bolt_diameter=_mm_to_m(args.bolt_diameter),
            bolt_allowable_stress=(
                args.bolt_allowable_stress * 1e6
                if args.bolt_allowable_stress is not None else None
            ),
            joint_separation_factor=args.joint_separation_factor,
            injector_face_od=_mm_to_m(args.injector_face_od),
            injector_face_thickness=_mm_to_m(args.injector_face_thickness),
            chamber_interface_length=_mm_to_m(args.interface_length),
        ),
        manufacturing=ManufacturingSpec(
            wall_thickness=_mm_to_m(args.wall_thickness),
            cad=cad,
            output_dir=build_dir,
            csv_points=args.n_csv,
            angular_points=args.n_angular,
            throat_insert=args.throat_insert,
            throat_insert_material=args.throat_insert_material,
            tolerance=_mm_to_m(args.tolerance),
            weld_allowance=_mm_to_m(args.weld_allowance),
            braze_allowance=_mm_to_m(args.braze_allowance),
        ),
        injector=InjectorSpec(
            type=args.injector,
            fuel_dp_fraction=args.fuel_injector_dp_fraction,
            oxidizer_dp_fraction=args.oxidizer_injector_dp_fraction,
            target_momentum_ratio=args.pintle_target_momentum_ratio,
            allow_infeasible=args.allow_infeasible_injector,
            fuel=PropellantFeedSpec(
                role='fuel',
                name=args.fuel,
                inlet_temperature=args.fuel_inlet_temperature,
                inlet_pressure=(
                    args.fuel_inlet_pressure * 1e5
                    if args.fuel_inlet_pressure is not None else None
                ),
            ),
            oxidizer=PropellantFeedSpec(
                role='oxidizer',
                name=args.oxidizer,
                inlet_temperature=args.oxidizer_inlet_temperature,
                inlet_pressure=(
                    args.oxidizer_inlet_pressure * 1e5
                    if args.oxidizer_inlet_pressure is not None else None
                ),
            ),
            geometry=PintleGeometrySpec(
                pintle_diameter=_mm_to_m(args.pintle_diameter),
                slot_count=args.pintle_slot_count,
                slot_aspect_ratio=args.pintle_slot_aspect_ratio,
                deflector_angle=args.pintle_deflector_angle,
                radial_stream=args.pintle_radial_stream,
                face_od=_mm_to_m(args.injector_face_od),
                face_thickness=_mm_to_m(args.injector_face_thickness),
            ),
            manufacturing=InjectorManufacturingSpec(
                min_feature=_mm_to_m(args.injector_min_feature),
            ),
            mechanical=PintleMechanicalSpec(
                bolt_count=args.bolt_count,
                bolt_circle_diameter=_mm_to_m(args.bolt_circle),
                bolt_hole_diameter=_mm_to_m(args.bolt_hole),
                faceplate_thickness=_mm_to_m(args.injector_face_thickness),
                faceplate_outer_diameter=_mm_to_m(args.injector_face_od),
                min_tool_diameter=_mm_to_m(args.injector_min_feature),
                min_corner_radius=_mm_to_m(args.injector_min_feature) / 2.0
                if args.injector_min_feature is not None else None,
                tolerance=_mm_to_m(args.tolerance),
            ),
            feed_system=FeedSystemSpec(
                architecture=args.feed_architecture,
                fuel=FeedLineSpec(
                    supply_pressure=(args.fuel_supply_pressure * 1e5
                                     if args.fuel_supply_pressure is not None
                                     else None),
                    flow_capacity=args.fuel_flow_capacity,
                    line_loss_fraction=args.fuel_line_loss_fraction,
                    manifold_loss_fraction=args.fuel_manifold_loss_fraction,
                    control_margin_fraction=args.feed_control_margin_fraction,
                    tank_pressure=(args.fuel_tank_pressure * 1e5
                                   if args.fuel_tank_pressure is not None
                                   else None),
                    npsh_required=(args.fuel_npsh_required * 1e5
                                   if args.fuel_npsh_required is not None
                                   else None),
                ),
                oxidizer=FeedLineSpec(
                    supply_pressure=(args.oxidizer_supply_pressure * 1e5
                                     if args.oxidizer_supply_pressure is not None
                                     else None),
                    flow_capacity=args.oxidizer_flow_capacity,
                    line_loss_fraction=args.oxidizer_line_loss_fraction,
                    manifold_loss_fraction=args.oxidizer_manifold_loss_fraction,
                    control_margin_fraction=args.feed_control_margin_fraction,
                    tank_pressure=(args.oxidizer_tank_pressure * 1e5
                                   if args.oxidizer_tank_pressure is not None
                                   else None),
                    npsh_required=(args.oxidizer_npsh_required * 1e5
                                   if args.oxidizer_npsh_required is not None
                                   else None),
                ),
            ),
            cad=args.injector_cad,
            cad_format=args.injector_cad_format,
        ),
        strict_gates=args.strict_gates,
    )

    result = design_nozzle_v2(design_input)
    _print_summary(
        result.propellant, result.contour, result.performance,
        Pc, Pa, result.input.Rt, result.input.epsilon,
    )
    _print_warnings(result.warnings)
    print("\n  ── V2 Physics Screening ───────────────────────────────")
    print(f"    Thermochemistry : {result.report_sections['thermochemistry']['source']}")
    print(f"    Heat flux max   : {result.report_sections['thermal']['q_max']/1e6:.2f} MW/m²")
    print(f"    Wall temp       : {result.report_sections['structural']['wall_temperature']:.1f} K")
    print(f"    Stress margin   : {result.report_sections['structural']['stress_margin']:.2f}")
    print(f"    Gate passed     : {result.gate_report.passed}")
    for name, path in result.files.items():
        print(f"  → {name}: {path}")

    params = {
        "Mode": result.input.mode,
        "Thermo": result.input.thermo.mode,
        "Propellant": result.propellant.name,
        "Pc [bar]": f"{Pc / 1e5:.2f}",
        "Pa [kPa]": f"{Pa / 1e3:.3f}",
        "Rt [mm]": f"{result.input.Rt * 1000:.2f}",
        "Epsilon (Ae/At)": f"{result.input.epsilon:.2f}",
        "Method": result.input.method,
        "Cooling": result.input.cooling.method,
        "Material": result.input.material.name,
    }
    perf_dict = {
        "Thrust [N]": f"{result.performance.thrust:.2f}",
        "Isp [s]": f"{result.performance.Isp:.1f}",
        "Mass flow [kg/s]": f"{result.performance.m_dot:.4f}",
        "Cf actual": f"{result.performance.Cf_actual:.4f}",
    }
    meta_path = write_metadata(
        build_dir, version=version, mode="batch-v2", params=params,
        performance=perf_dict, warnings=result.warnings,
        gate_report=result.gate_report.to_dict(),
        files=[path.name for path in result.files.values()],
    )
    print(f"  → metadata: {meta_path}")
    print(f"\n  📁 Build v{version:03d}: {build_dir}")
    print("\n  Done.\n")


def run_batch(args):
    """Non-interactive mode: all params from argparse."""
    if _should_use_v2(args):
        run_batch_v2(args)
        return

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
    elif args.method == 'rao_variational_moc':
        contour = bell_nozzle_contour(Rt, epsilon, method='rao_variational_moc',
                                       length_pct=length_pct,
                                       gamma=prop.gamma,
                                       pa_over_p0=args.pa_over_p0,
                                       rao_moc_n_control=args.rao_moc_n_control,
                                       rao_moc_n_kernel=args.rao_moc_n_kernel,
                                       rao_moc_max_nfev=args.rao_moc_max_nfev,
                                       rao_moc_evaluate_moc=not args.rao_moc_skip_moc)
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


def run_benchmark_cli(args):
    """Run literature benchmark mode."""
    _header()
    from raosim.benchmarks import list_benchmark_cases, run_benchmark

    methods = (
        ['bezier', 'moc', 'rao']
        if args.benchmark_method == 'all'
        else [args.benchmark_method]
    )
    report_base = Path(args.benchmark_report) if args.benchmark_report else None
    results = []

    print(f"  Benchmark case: {args.benchmark_case}")
    print(f"  Available cases: {', '.join(list_benchmark_cases())}")
    print()

    for method in methods:
        report_path = _benchmark_report_path(report_base, method, len(methods) > 1)
        result = run_benchmark(
            args.benchmark_case,
            method,
            report_path=report_path,
        )
        results.append(result)
        counts = _status_counts(result['metrics'])
        print(f"  [{result['overall_status'].upper():>6}] {method}")
        print(f"    JSON: {result['report_paths']['json']}")
        print(f"    MD:   {result['report_paths']['markdown']}")
        print(
            "    Metrics: "
            + ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        )
        if result.get('warnings'):
            print(f"    Solver warnings: {len(result['warnings'])}")
        print()

    if args.benchmark_strict:
        failed = [r for r in results if r['overall_status'] != 'pass']
        if failed:
            print("  ✗ Benchmark strict mode failed:")
            for result in failed:
                print(f"    • {result['method']}: {result['overall_status']}")
            sys.exit(2)

    print("  Done.\n")


def _benchmark_report_path(base: Path | None, method: str, multi: bool) -> Path | None:
    if base is None or not multi or not base.suffix:
        return base
    return base.with_name(f"{base.stem}_{method}{base.suffix}")


def _status_counts(metrics: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for metric in metrics:
        status = metric.get('status', 'unknown')
        counts[status] = counts.get(status, 0) + 1
    return counts


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

    if args.benchmark_case:
        run_benchmark_cli(args)
    elif args.sweep:
        run_sweep(args)
    elif is_batch(args):
        run_batch(args)
    else:
        run_interactive()


if __name__ == "__main__":
    main()
