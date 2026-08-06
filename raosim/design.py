"""
design.py - High-level design-gated nozzle workflow.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from raosim.cea import (
    PinnedChamberState,
    THERMO_CEA_FROZEN,
    THERMO_CONSTANT_GAMMA,
    THERMO_PINNED_CHAMBER,
    ThermochemistryResult,
    propellant_from_request,
    resolve_thermochemistry,
)
from raosim.chamber_geometry import (
    auto_shoulder_factor,
    chamber_contour,
    full_engine_contour,
    thrust_chamber_geometry_checks,
)
from raosim.coolants import canonical_coolant_name
from raosim.engine import EnginePerformance, compute_engine_performance
from raosim.export import export_csv, export_step, export_stl, package_ipt_request
from raosim.gas_dynamics import (
    expansion_ratio_from_pressure,
    isentropic_pressure_ratio,
    mach_from_area_ratio,
    thrust_coefficient,
)
from raosim.injector import InjectorSpec, evaluate_pintle_injector
from raosim.interface import (
    resolve_bolted_interface_geometry,
    screen_injector_chamber_interface,
)
from raosim.model_registry import audit_model_registry, model_provenance_dict
from raosim.nozzle_geometry import bell_nozzle_contour
from raosim.physics import (
    bartz_heat_flux,
    boundary_layer_displacement,
    regenerative_cooling_screen,
    structural_screen,
)
from raosim.propellants import Propellant
from raosim.release_readiness import (
    evaluate_release_readiness,
    load_evidence_manifest,
)
from raosim.spray_coupling import (
    SprayCStarCouplingSpec,
    solve_spray_cstar_fixed_point,
)
from raosim.throat_geometry import (
    REPOSITORY_UPSTREAM_RADIUS_RATIO_EXTENSION_BOUNDS,
    SP8120_UPSTREAM_RADIUS_RATIO_BOUNDS,
    ThroatGeometrySpec,
    throat_discharge_coefficient_hall,
    upstream_radius_ratio_for_discharge_coefficient,
)
from raosim.validation import (
    DesignGateReport,
    add_contour_reliability_metadata,
    evaluate_design_gates,
)


DESIGN_MODE_PRELIMINARY = "preliminary"
DESIGN_MODE_VALIDATED = "validated"
DESIGN_MODES = {DESIGN_MODE_PRELIMINARY, DESIGN_MODE_VALIDATED}

CAD_NONE = "none"
CAD_STEP = "step"
CAD_STL = "stl"
CAD_BOTH = "both"
CAD_IPT = "ipt"
CAD_MODES_V2 = {CAD_NONE, CAD_STEP, CAD_STL, CAD_BOTH}


@dataclass
class NozzleDesignRequest:
    propellant_name: str | None
    Pc: float
    Pa: float = 101_325.0
    Rt: float | None = None
    target_thrust: float | None = None
    epsilon: float = 10.0
    method: str = "bezier"
    length_pct: float = 80.0
    theta_n: float | None = None
    theta_e: float | None = None
    mixture_ratio: float | None = None
    use_cea: bool = False
    oxidizer: str | None = None
    fuel: str | None = None
    eta_Isp: float = 0.95
    wall_thickness: float | None = None
    flange_od: float | None = None
    flange_length: float | None = None
    contraction_ratio: float | None = None
    cad: str = "none"
    output_dir: Path | None = None
    csv_points: int = 301
    angular_points: int = 64
    strict_gates: bool = False


@dataclass
class ThermoSpec:
    mode: str = THERMO_CONSTANT_GAMMA
    propellant_name: str | None = None
    oxidizer: str | None = None
    fuel: str | None = None
    mixture_ratio: float | None = None
    # Legacy lumped Isp efficiency.  When the explicit split below is
    # omitted, the existing thermochemistry-provider convention is retained.
    # Supplying eta_cstar and/or eta_CF makes those values authoritative and
    # eta_Isp is recomputed from the resolved split.
    eta_Isp: float = 0.95
    # Expansion physics is separate from the chamber-state provider.  The
    # variable-cp option requires a configuration-controlled fixed-composition
    # cp(T) JSON table and is currently limited to Bezier geometry.
    expansion_model: str = "constant_gamma"
    frozen_gas_table: Path | None = None
    # Appended after the legacy fields so positional ThermoSpec callers retain
    # their historical argument ordering.
    eta_cstar: float | None = None
    eta_CF: float | None = None
    # Host-only immutable state used to align a traditional re-evaluation with
    # the exact gamma/Tc/R/c-star values retained by a solved MDO EngineState.
    # This remains preliminary parity evidence, not an independent CEA rerun.
    pinned_chamber_state: PinnedChamberState | None = None


@dataclass
class MissionAmbientSpec:
    Pa: float = 101_325.0
    altitude_schedule_m: list[float] = field(default_factory=list)
    pressure_schedule_pa: list[float] = field(default_factory=list)

    @property
    def design_pressure(self) -> float:
        if self.pressure_schedule_pa:
            return float(min(self.pressure_schedule_pa))
        return float(self.Pa)


@dataclass
class HostRaoSolverSpec:
    """Host-only controls for the experimental Rao variational/MOC analysis.

    These values configure the existing auditable host solve used for
    post-analysis.  They do not replace the fixed-topology chart/Bézier wall in
    the differentiable MDO core.
    """

    n_control: int = 12
    n_kernel: int = 12
    max_nfev: int = 200
    evaluate_moc: bool = True
    theta_n_guess_deg: float = 30.0
    starting_line_method: str = "kliegel_levine"
    solver_backend: str = "jax"
    wall_method: str = "coupled"
    kernel_d_fraction_max: float | None = None
    physics_weight: float | None = None


@dataclass
class CoolingSpec:
    method: str = "none"
    coolant: str | None = None
    channel_count: int | None = None
    channel_width: float | None = None
    channel_height: float | None = None
    # Arithmetic mean internal roughness [m]. Zero retains the ideal
    # smooth-wall screening branch; positive values use Swamee-Jain for
    # turbulent Darcy loss.
    channel_roughness: float = 0.0
    coolant_mass_flow: float | None = None
    # None → the COOLANT_PROPERTIES table value for the named coolant
    # is authoritative (e.g. RP-1 ≈ 2010 J/kg·K).  Set explicitly only
    # to override the table with a CEA/measured cp.  (Was 3500, which
    # silently overrode every named coolant's table cp — a latent bug.)
    coolant_cp: float | None = None
    # None is resolved centrally by coolant identity: methane 120 K, LH2
    # 25 K, and 300 K for the current non-cryogenic preliminary defaults.
    coolant_inlet_temperature: float | None = None
    # Absolute jacket pressure immediately before the coolant leaves the
    # cooling passages for the injector/manifold.  When omitted, high-level
    # workflows derive the boundary from InjectorSpec.fuel_dp_fraction when
    # the coolant is the cycle fuel.
    coolant_outlet_pressure: float | None = None
    # Deprecated compatibility field for direct low-level cooling callers.
    # design_nozzle_v2 ignores user-supplied values here and derives the
    # fuel-side boundary from InjectorSpec.fuel_dp_fraction instead.
    injector_pressure_drop: float = 0.0
    max_wall_temperature: float = 950.0
    # Optional coolant transport properties (override the built-in
    # COOLANT_PROPERTIES table keyed by ``coolant`` name).  Supply
    # CEA/measured values for accuracy; used by the Sieder-Tate solve.
    coolant_density: float | None = None        # kg/m^3
    coolant_viscosity: float | None = None      # Pa.s (disables Andrade T-model)
    coolant_conductivity: float | None = None   # W/(m.K)
    # Optional chemistry/phase stability limit at the coolant-side wall.
    # None uses a documented coolant-specific limit where available
    # (currently RP-1/kerosene coking); nonpositive disables that check.
    coolant_wall_temperature_limit: float | None = None
    # Thermophysical backend. ``auto`` uses CoolProp for methane/LH2 when
    # installed and otherwise retains the explicit constant-property screen.
    coolant_property_backend: str = "auto"
    # Optional full inlet-plenum / channel / outlet-plenum hydraulic graph.
    hydraulic_network: bool = False
    ports_per_manifold: int = 4
    port_area_ratio: float = 1.0
    port_diameter: float | None = None
    plenum_area_ratio: float = 2.0
    plenum_hydraulic_diameter: float | None = None
    port_loss_coefficient: float = 1.5
    channel_entry_loss_coefficient: float = 0.5
    channel_exit_loss_coefficient: float = 1.0
    # Participating-gas radiation. ``none`` preserves Bartz convection only;
    # ``leccese_gray`` uses the documented CH4/H2 screening preset.
    radiation_model: str = "none"
    radiation_propellant_family: str | None = None
    radiation_path_length: float | None = None
    radiation_wall_emissivity: float = 1.0
    radiation_bands: tuple[dict, ...] = ()
    # Real-fluid boiling / CHF diagnostics. The gate is separate because the
    # implemented Zuber CHF is a conservative screening reference.
    boiling_chf: bool = False
    gate_chf: bool = False
    # Optional fuel branch that leaves the common, upstream fuel-pump stream
    # and bypasses the regenerative jacket for separate wall-film injection.
    # For a fuel-as-coolant topology the injector integration enforces
    #
    #   coolant_mass_flow + fuel_film_mass_flow = cycle fuel mass flow.
    #
    # Zero preserves the historical direct jacket-to-injector topology.  This
    # field records the split; film-slot/orifice hardware remains outside the
    # traditional injector model. Appended to preserve positional callers.
    fuel_film_mass_flow: float = 0.0


@dataclass
class MaterialSpec:
    name: str = "Inconel 718"
    yield_strength: float = 900e6
    conductivity: float = 16.0
    max_temperature: float = 1250.0
    max_heat_flux: float = 25e6
    # Extra elastic/thermal properties needed by the SP-125 coaxial-shell
    # combined-stress check (eq. 4-31: pressure differential + thermal
    # stress).  Default None keeps every existing caller back-compatible;
    # populated by :meth:`from_catalog`.
    elastic_modulus: float | None = None     # E   [Pa]
    thermal_expansion: float | None = None   # a   [1/K]
    poisson_ratio: float | None = None       # v   [-]
    density: float | None = None             # rho [kg/m^3]
    category: str | None = None
    # Coffin-Manson strain-life coefficients (low-cycle fatigue screen).
    fatigue_strength_coeff: float | None = None    # sigma_f' [Pa]
    fatigue_strength_exp: float | None = None      # b
    fatigue_ductility_coeff: float | None = None   # eps_f'
    fatigue_ductility_exp: float | None = None     # c
    fatigue_source: str | None = None
    fatigue_data_temperature: float | None = None
    fatigue_design_qualified: bool = False
    fatigue_screening_gate: bool = False
    fatigue_total_strain_curves: tuple[
        tuple[float, float, float, float, str], ...
    ] = ()
    cyclic_stress_strain_curves: tuple[
        tuple[float, float, float], ...
    ] = ()
    cyclic_stress_strain_source: str | None = None
    # Required yield/combined-stress ratio.  Kept explicit so an MDO
    # post-FOS allowable can be mapped back to yield strength without silently
    # applying the historical hard-coded 1.5 twice.
    structural_fos: float = 1.5
    # --- structural closeout (jacket) ------------------------------------- #
    # NASA SP-8087 sec. 2.1.3.1: "Hardenable materials often are used for
    # jacket designs, where, after brazing, the strength can be increased
    # considerably by agehardening."  A regeneratively cooled wall is normally
    # a soft high-conductivity liner inside a strong jacket, so the jacket
    # alloy is a separate field rather than an inherited one.  ``None`` uses
    # the repository's standard jacket/structure entry.
    jacket_material: str | None = None
    # SP-8087 sec. 2.1.3 quotes yield factors of safety of 1.0-1.32 (and
    # ultimate 1.3-1.8).  The conservative end of the yield band is used to
    # size the jacket against the SP-125 p.109 outer-shell hoop stress.
    closeout_structural_fos: float = 1.32

    @classmethod
    def from_catalog(cls, name: str) -> "MaterialSpec":
        """Build a fully-populated spec from the :mod:`raosim.materials`
        catalog (literature-grounded k, S_y, T_max, E, a, v, rho, and the
        Coffin-Manson fatigue coefficients)."""
        from raosim.materials import get_material
        m = get_material(name)
        return cls(
            name=m.name, yield_strength=m.yield_strength,
            conductivity=m.conductivity, max_temperature=m.max_temperature,
            max_heat_flux=m.max_heat_flux, elastic_modulus=m.elastic_modulus,
            thermal_expansion=m.thermal_expansion,
            poisson_ratio=m.poisson_ratio, density=m.density,
            category=m.category,
            fatigue_strength_coeff=m.fatigue_strength_coeff,
            fatigue_strength_exp=m.fatigue_strength_exp,
            fatigue_ductility_coeff=m.fatigue_ductility_coeff,
            fatigue_ductility_exp=m.fatigue_ductility_exp,
            fatigue_source=m.fatigue_source,
            fatigue_data_temperature=m.fatigue_data_temperature,
            fatigue_design_qualified=m.fatigue_design_qualified,
            fatigue_screening_gate=m.fatigue_screening_gate,
            fatigue_total_strain_curves=m.fatigue_total_strain_curves,
            cyclic_stress_strain_curves=m.cyclic_stress_strain_curves,
            cyclic_stress_strain_source=m.cyclic_stress_strain_source,
        )


@dataclass
class InterfaceSpec:
    flange_od: float | None = None
    flange_length: float | None = None
    bolt_count: int | None = None
    bolt_circle_diameter: float | None = None
    bolt_hole_diameter: float | None = None
    bolt_diameter: float | None = None
    bolt_allowable_stress: float | None = None
    joint_separation_factor: float = 1.5
    injector_face_od: float | None = None
    injector_face_thickness: float | None = None
    chamber_interface_length: float | None = None


@dataclass
class ManufacturingSpec:
    wall_thickness: float | None = None
    cad: str = CAD_NONE
    output_dir: Path | None = None
    csv_points: int = 301
    angular_points: int = 64
    throat_insert: bool = False
    throat_insert_material: str | None = None
    tolerance: float | None = None
    weld_allowance: float | None = None
    braze_allowance: float | None = None


@dataclass
class DesignInput:
    thermo: ThermoSpec
    Pc: float
    Rt: float | None = None
    target_thrust: float | None = None
    epsilon: float | None = 10.0
    method: str = "bezier"
    mode: str = DESIGN_MODE_PRELIMINARY
    length_pct: float = 80.0
    theta_n: float | None = None
    theta_e: float | None = None
    contraction_ratio: float | None = None
    L_star: float | None = None
    shoulder_radius_factor: float | None = None
    shoulder_fill_fraction: float = 0.8
    minimum_cylindrical_length: float | None = None
    throat_cd_target: float | None = None
    allow_throat_radius_extension: bool = False
    throat_geometry: ThroatGeometrySpec = field(default_factory=ThroatGeometrySpec)
    ambient: MissionAmbientSpec = field(default_factory=MissionAmbientSpec)
    cooling: CoolingSpec = field(default_factory=CoolingSpec)
    material: MaterialSpec = field(default_factory=MaterialSpec)
    interface: InterfaceSpec = field(default_factory=InterfaceSpec)
    manufacturing: ManufacturingSpec = field(default_factory=ManufacturingSpec)
    injector: "InjectorSpec" = field(default_factory=lambda: InjectorSpec())
    # Opt-in closure of eta_cstar -> cycle mdot -> injector vaporization ->
    # eta_cstar.  Mixing and chemical-completion efficiencies are never
    # inferred by the injector screen and must be supplied explicitly.
    spray_cstar_coupling: SprayCStarCouplingSpec = field(
        default_factory=SprayCStarCouplingSpec
    )
    # Physical release remains separate from numerical/design-gate success.
    # Supplying a manifest makes its traceable evidence part of the report;
    # require_release_evidence turns missing/failed evidence into a hard gate.
    configuration_id: str | None = None
    release_evidence_manifest: Path | None = None
    require_release_evidence: bool = False
    strict_gates: bool = False
    # Appended so existing positional DesignInput callers retain their ordering.
    host_rao_solver: HostRaoSolverSpec = field(
        default_factory=HostRaoSolverSpec
    )


@dataclass
class ValidatedDesignResult:
    input: DesignInput
    thermochemistry: ThermochemistryResult
    propellant: Propellant
    contour: dict
    performance: EnginePerformance
    gate_report: DesignGateReport
    report_sections: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    files: dict[str, Path] = field(default_factory=dict)

    @property
    def design_status(self) -> str:
        return str(self.contour.get("design_status", "unknown_geometry"))

    @property
    def validated(self) -> bool:
        return self.input.mode == DESIGN_MODE_VALIDATED and self.gate_report.passed

    @property
    def hardware_qualified(self) -> bool:
        """Hardware qualification can never be asserted by this workflow."""

        return False

    def to_dict(self) -> dict:
        return _json_ready({
            "mode": self.input.mode,
            "validated": self.validated,
            "software_validated": self.validated,
            "hardware_qualified": False,
            "design_status": self.design_status,
            "warnings": self.warnings,
            "files": {key: str(value) for key, value in self.files.items()},
            "gate_report": self.gate_report.to_dict(),
            "report_sections": self.report_sections,
            "performance": {
                "thrust": self.performance.thrust,
                "Isp": self.performance.Isp,
                "m_dot": self.performance.m_dot,
                "Cf_actual": self.performance.Cf_actual,
                "Cf_ideal": self.performance.Cf_ideal,
                "c_star": self.performance.c_star,
                "c_star_effective": self.performance.c_star_effective,
                "Pe": self.performance.Pe,
                "Me": self.performance.Me,
                "expansion_model": self.performance.expansion_model,
                "gamma_throat": self.performance.gamma_throat,
                "gamma_exit": self.performance.gamma_exit,
                "exit_temperature": self.performance.exit_temperature,
                "frozen_flow_fingerprint": (
                    self.performance.frozen_flow_fingerprint
                ),
            },
        })


@dataclass
class DesignResult:
    request: NozzleDesignRequest
    propellant: Propellant
    contour: dict
    performance: EnginePerformance
    gate_report: DesignGateReport
    warnings: list[str] = field(default_factory=list)
    files: dict[str, Path] = field(default_factory=dict)

    @property
    def design_status(self) -> str:
        return str(self.contour.get("design_status", "unknown_geometry"))

    def to_dict(self) -> dict:
        return {
            "design_status": self.design_status,
            "warnings": self.warnings,
            "files": {key: str(value) for key, value in self.files.items()},
            "gate_report": self.gate_report.to_dict(),
            "performance": {
                "thrust": self.performance.thrust,
                "Isp": self.performance.Isp,
                "m_dot": self.performance.m_dot,
                "Cf_actual": self.performance.Cf_actual,
                "Pe": self.performance.Pe,
                "Me": self.performance.Me,
            },
        }


@dataclass(frozen=True)
class SprayRegenIterationPayload:
    """State re-solved at one spray/c-star/regen mass-flow iterate."""

    injector: Any
    thermal: dict[str, Any]
    cooling: dict[str, Any]
    total_mass_flow: float
    fuel_mass_flow: float
    oxidizer_mass_flow: float
    coolant_mass_flow: float
    fuel_film_mass_flow: float = 0.0

    def coupling_summary(self) -> dict[str, Any]:
        regenerative = self.cooling.get("method") == "regenerative"
        feed_lines = {}
        ledger = getattr(self.injector, "feed_system", None)
        if ledger is not None:
            for role, line in ledger.lines.items():
                feed_lines[role] = {
                    "mass_flow_kg_s": self.injector.streams[role].mdot,
                    "required_outlet_pressure_pa": line.required_outlet_pressure,
                    "regen_loss_pa": line.regen_loss,
                    "volumetric_flow_m3_s": line.volumetric_flow,
                    "required_pump_head_m": line.required_pump_head,
                    "ideal_pump_power_w": line.ideal_pump_power,
                    "status": line.status,
                }
        return {
            "total_mass_flow_kg_s": self.total_mass_flow,
            "fuel_mass_flow_kg_s": self.fuel_mass_flow,
            "oxidizer_mass_flow_kg_s": self.oxidizer_mass_flow,
            "coolant_mass_flow_kg_s": self.coolant_mass_flow,
            "fuel_film_mass_flow_kg_s": self.fuel_film_mass_flow,
            "regen_fuel_relative_flow_error": (
                abs(
                    self.coolant_mass_flow
                    + self.fuel_film_mass_flow
                    - self.fuel_mass_flow
                )
                / max(self.fuel_mass_flow, 1.0e-30)
                if regenerative else None
            ),
            "fuel_flow_topology": (
                "common_upstream_pump_then_regen_and_film_split"
                if regenerative and self.fuel_film_mass_flow > 0.0
                else (
                    "direct_regen_jacket_to_injector"
                    if regenerative else None
                )
            ),
            "outer_loop_scope": (
                "spray_cycle_regen_wall_feed_and_pump_duty"
                if regenerative else "injector_and_cycle_mass_flow_no_regen"
            ),
            "cooling_margin": self.cooling.get("cooling_margin"),
            "coolant_outlet_temperature_k": self.cooling.get(
                "coolant_outlet_temperature"
            ),
            "coolant_pressure_drop_pa": self.cooling.get(
                "coolant_pressure_drop"
            ),
            "estimated_wall_temperature_k": self.cooling.get(
                "estimated_wall_temperature",
                self.cooling.get("peak_gas_side_wall_temperature"),
            ),
            "feed_and_pump_duty_by_role": feed_lines,
        }


def design_nozzle_v2(input: DesignInput) -> ValidatedDesignResult:
    """Generate a physics-screened nozzle design from the strict v2 schema."""
    _validate_design_input(input)
    design_ambient_pressure = input.ambient.design_pressure
    require_cea = input.mode == DESIGN_MODE_VALIDATED

    thermo = resolve_thermochemistry(
        thermo_mode=input.thermo.mode,
        propellant_name=input.thermo.propellant_name,
        Pc=input.Pc,
        mixture_ratio=input.thermo.mixture_ratio,
        oxidizer=input.thermo.oxidizer,
        fuel=input.thermo.fuel,
        eta_Isp=input.thermo.eta_Isp,
        eta_cstar=input.thermo.eta_cstar,
        eta_CF=input.thermo.eta_CF,
        epsilon=input.epsilon,
        require_cea=require_cea,
        pinned_chamber_state=input.thermo.pinned_chamber_state,
    )
    prop = thermo.propellant
    warnings = list(thermo.warnings)
    frozen_gas = None
    if input.thermo.expansion_model == "frozen_variable_cp":
        from raosim.frozen_flow import load_frozen_gas_table

        frozen_gas = load_frozen_gas_table(input.thermo.frozen_gas_table)
        requested_mixture_ratio = (
            input.thermo.mixture_ratio
            if input.thermo.mixture_ratio is not None
            else prop.OF
        )
        if (
            frozen_gas.freeze_basis == "chamber_equilibrium_snapshot"
            and requested_mixture_ratio is not None
            and not math.isclose(
                float(frozen_gas.mixture_ratio),
                float(requested_mixture_ratio),
                rel_tol=1.0e-9,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(
                "frozen gas table mixture_ratio does not match the requested "
                "chamber composition state"
            )
        thermo.exit_state.update({
            "expansion_model": "frozen_variable_cp_q1d",
            "frozen_gas_fingerprint_sha256": frozen_gas.fingerprint_sha256,
            "frozen_gas_input_artifact_sha256": (
                frozen_gas.input_artifact_sha256
            ),
            "composition_mass_fractions": dict(
                frozen_gas.composition_mass_fractions
            ),
            "property_source": frozen_gas.source,
            "property_table": frozen_gas.as_dict(),
            "equilibrium_chemistry": False,
            "moc_or_rao_characteristics": False,
        })
        warnings.append(
            "Using thermally-perfect fixed-composition quasi-1-D expansion. "
            "MOC/Rao characteristics are disabled; Bartz, boundary-layer, and "
            "Hall-Cd outputs remain separately failed screening gates until "
            "profile-aware variable-property models are implemented."
        )

    epsilon = input.epsilon
    if epsilon is None:
        if frozen_gas is not None:
            from raosim.frozen_flow import expansion_ratio_from_pressure_frozen

            epsilon, _ = expansion_ratio_from_pressure_frozen(
                frozen_gas,
                chamber_pressure_pa=input.Pc,
                chamber_temperature_k=prop.Tc,
                exit_pressure_pa=input.ambient.design_pressure,
            )
        else:
            epsilon, _ = expansion_ratio_from_pressure(
                input.Pc, input.ambient.design_pressure, prop.gamma
            )
        input.epsilon = epsilon
        warnings.append(
            f"Sized epsilon from Pc/design Pa: epsilon = {epsilon:.3f}."
        )

    Rt = input.Rt
    rt_sized_from_target = Rt is None and input.target_thrust is not None
    if Rt is not None and input.target_thrust is not None:
        warnings.append("Both Rt and target_thrust supplied; explicit Rt is used.")
    if Rt is None:
        if input.target_thrust is None:
            raise ValueError("Either Rt or target_thrust must be provided.")
        Rt = throat_radius_for_target_thrust(
            input.target_thrust, input.Pc, input.ambient.design_pressure,
            float(epsilon), prop, frozen_gas=frozen_gas,
        )
        input.Rt = Rt
        warnings.append(f"Sized Rt from target thrust: Rt = {Rt * 1000:.3f} mm.")

    contour = _build_v2_contour(input, Rt, float(epsilon), prop)
    interface_resolution = _apply_pintle_interface_resolution(input, contour)
    if (
        input.L_star is None
        or input.contraction_ratio is None
        or input.minimum_cylindrical_length is None
    ):
        warnings.append(
            "Chamber sizing used one or more geometric placeholders "
            "(L*=1.0 m, contraction ratio=2.5, minimum cylinder=1e-6 m). "
            "The chamber shoulder fillet is auto-derived from geometric "
            "closure when not supplied. Select injector-, mixing-, packaging-, "
            "pressure-, mixture-ratio-, and duty-informed values."
        )
    performance = compute_engine_performance(
        input.Pc, design_ambient_pressure, Rt, float(epsilon), prop,
        frozen_gas=frozen_gas,
    )
    if performance.frozen_flow is not None:
        add_contour_reliability_metadata(
            contour,
            input.method,
            prop.gamma,
            frozen_expansion=performance.frozen_flow,
        )
    target_error = None
    if input.target_thrust is not None:
        target_error = (
            performance.thrust - input.target_thrust
        ) / input.target_thrust
    thrust_closure = {
        "sizing_basis": (
            "quasi_1d_thermally_perfect_frozen_composition_delivered_cf"
            if frozen_gas is not None
            else "quasi_1d_calorically_perfect_delivered_cf"
        ),
        "rt_sized_from_target_thrust": bool(rt_sized_from_target),
        "target_thrust_N": input.target_thrust,
        "calculated_thrust_N": performance.thrust,
        "relative_target_error": target_error,
        "quasi_1d_Cf_ideal": performance.Cf_ideal,
        "quasi_1d_Cf_delivered": performance.Cf_actual,
        "contour_audit_Cf": None,
        "contour_audit_basis": "unavailable_for_geometry_only_workflow",
        "throat_discharge_coefficient_applied_to_mass_flow": False,
        "throat_discharge_coefficient_note": (
            "The Hall/SP-8120 throat Cd is a reported inviscid curvature "
            "screen. It is not folded into mdot=Pc*At/cstar or target-thrust "
            "sizing; doing so requires a consistently calibrated delivered "
            "mass-flow/performance model."
        ),
    }

    boundary_layer = boundary_layer_displacement(contour, input.Pc, prop)
    thermal = bartz_heat_flux(contour, input.Pc, prop)
    mixture_ratio = input.thermo.mixture_ratio
    if mixture_ratio is None:
        mixture_ratio = float(getattr(prop, "OF", 0.0) or 0.0)
    # Pass prop + Pc so the cooling screen runs the real coupled
    # Sieder-Tate / 1-D wall-conduction solve (gas side = full Bartz).
    cooling_input, cooling_boundary_warnings = _cooling_at_cycle_mass_flow(
        input,
        total_mass_flow=performance.m_dot,
        mixture_ratio=mixture_ratio,
    )
    if not (
        input.spray_cstar_coupling.enabled
        and input.cooling.method == "regenerative"
    ):
        warnings.extend(cooling_boundary_warnings)
    cooling = regenerative_cooling_screen(
        thermal, contour, cooling_input, input.material,
        input.manufacturing.wall_thickness, prop, input.Pc,
    )
    injector_result = None
    spray_coupling_result = None
    if input.injector.type == "pintle":
        if mixture_ratio <= 0.0:
            raise ValueError(
                "Pintle injector sizing requires a positive mixture ratio; "
                "set thermo.mixture_ratio or select a propellant with nominal O/F."
            )
        oxidizer_name, fuel_name = _thermo_feed_names(input.thermo)

        def evaluate_injector_at_mass_flow(
            total_mdot: float,
            gas_propellant: Propellant,
            *,
            iteration_cooling: CoolingSpec = cooling_input,
            iteration_cooling_result: dict[str, Any] = cooling,
        ):
            mdot_fuel = total_mdot / (1.0 + mixture_ratio)
            mdot_oxidizer = mixture_ratio * mdot_fuel
            return evaluate_pintle_injector(
                input.injector,
                mdot_fuel=mdot_fuel,
                mdot_oxidizer=mdot_oxidizer,
                Pc=input.Pc,
                mixture_ratio=mixture_ratio,
                chamber_radius=float(contour["chamber"]["Rc"]),
                chamber_length=float(
                    contour["chamber"]["injector_to_throat_length"]
                ),
                gamma=gas_propellant.gamma,
                Tc=gas_propellant.Tc,
                R_gas=gas_propellant.R_gas,
                fuel_name=fuel_name,
                oxidizer_name=oxidizer_name,
                cooling=iteration_cooling,
                cooling_result=iteration_cooling_result,
            )

        if input.spray_cstar_coupling.enabled:
            ideal_cstar = float(performance.c_star)

            def coupling_evaluator(eta_cstar: float, total_mdot: float):
                trial_prop = _propellant_with_eta_cstar(prop, eta_cstar)
                trial_thermal = bartz_heat_flux(
                    contour, input.Pc, trial_prop
                )
                trial_cooling_input, _ = _cooling_at_cycle_mass_flow(
                    input,
                    total_mass_flow=total_mdot,
                    mixture_ratio=mixture_ratio,
                )
                trial_cooling = regenerative_cooling_screen(
                    trial_thermal,
                    contour,
                    trial_cooling_input,
                    input.material,
                    input.manufacturing.wall_thickness,
                    trial_prop,
                    input.Pc,
                )
                trial_injector = evaluate_injector_at_mass_flow(
                    total_mdot,
                    trial_prop,
                    iteration_cooling=trial_cooling_input,
                    iteration_cooling_result=trial_cooling,
                )
                if trial_injector.atomization is None:
                    raise RuntimeError(
                        "spray/c-star coupling requires an injector atomization result"
                    )
                fuel_mdot = total_mdot / (1.0 + mixture_ratio)
                state = SprayRegenIterationPayload(
                    injector=trial_injector,
                    thermal=trial_thermal,
                    cooling=trial_cooling,
                    total_mass_flow=total_mdot,
                    fuel_mass_flow=fuel_mdot,
                    oxidizer_mass_flow=mixture_ratio * fuel_mdot,
                    coolant_mass_flow=float(
                        trial_cooling_input.coolant_mass_flow or 0.0
                    ),
                    fuel_film_mass_flow=float(
                        trial_cooling_input.fuel_film_mass_flow or 0.0
                    ),
                )
                return trial_injector.atomization, state

            spray_coupling_result = solve_spray_cstar_fixed_point(
                input.spray_cstar_coupling,
                initial_eta_cstar=float(prop.eta_cstar),
                ideal_cstar=ideal_cstar,
                chamber_pressure=input.Pc,
                throat_area=math.pi * Rt ** 2,
                evaluator=coupling_evaluator,
            )
            prop = _propellant_with_eta_cstar(
                prop, spray_coupling_result.eta_cstar
            )
            thermo.propellant = prop
            thermo.chamber_state["eta_cstar_coupled"] = prop.eta_cstar
            thermo.chamber_state["c_star_effective"] = prop.c_star_effective
            performance = compute_engine_performance(
                input.Pc, design_ambient_pressure, Rt, float(epsilon), prop,
                frozen_gas=frozen_gas,
            )
            final_state = spray_coupling_result.payload
            if not isinstance(final_state, SprayRegenIterationPayload):
                raise RuntimeError(
                    "spray/c-star evaluator returned an unexpected final state"
                )
            injector_result = final_state.injector
            thermal = final_state.thermal
            cooling = final_state.cooling
            _, final_cooling_warnings = _cooling_at_cycle_mass_flow(
                input,
                total_mass_flow=spray_coupling_result.required_mass_flow,
                mixture_ratio=mixture_ratio,
            )
            warnings.extend(final_cooling_warnings)
        else:
            injector_result = evaluate_injector_at_mass_flow(
                performance.m_dot, prop
            )
        interface_resolution = _apply_pintle_interface_resolution(
            input, contour, injector_result=injector_result
        )
    gate_flange_od = (
        input.interface.flange_od
        if (
            input.manufacturing.wall_thickness is not None
            or input.manufacturing.cad.lower() != CAD_NONE
        )
        else None
    )
    gate_flange_length = (
        input.interface.flange_length
        if gate_flange_od is not None else None
    )
    gate_report = evaluate_design_gates(
        contour,
        input.Pc,
        design_ambient_pressure,
        prop.gamma,
        frozen_expansion=performance.frozen_flow,
        wall_thickness=input.manufacturing.wall_thickness,
        flange_od=gate_flange_od,
        flange_length=gate_flange_length,
    )
    if injector_result is not None:
        for gate in injector_result.gates:
            gate_report.add(
                "injector",
                gate.name,
                gate.status != "fail",
                value=gate.status,
                limit="status != fail",
                message=gate.detail,
            )
    structural = structural_screen(
        contour, input.Pc, design_ambient_pressure, prop, input.material,
        input.manufacturing.wall_thickness, thermal, cooling,
        channel_width=getattr(input.cooling, "channel_width", None),
    )
    injector_interface = _injector_interface_screen(input, contour)
    for gate in injector_interface.gates:
        gate_report.add(
            "interface",
            gate.name,
            gate.status != "fail",
            value=gate.value if gate.value is not None else gate.status,
            limit=gate.limit if gate.limit is not None else "status != fail",
            message=gate.detail,
        )
    hardware_mass = _hardware_mass_section(
        input, contour, cooling_input, cooling, interface_resolution,
        injector_result,
    )
    cad_readiness = _cad_readiness(input, contour, gate_report)
    benchmark_status = _benchmark_status(input.method, contour)

    if input.release_evidence_manifest is not None:
        release_readiness = load_evidence_manifest(
            input.release_evidence_manifest,
            expected_target="engine",
            expected_configuration_id=input.configuration_id,
        )
    else:
        release_readiness = evaluate_release_readiness(target="engine")
    if input.require_release_evidence:
        # This is intentionally evaluated before any design/CAD artifacts are
        # written.  Passing it means the configured evidence set is complete;
        # it still does not confer hardware qualification.
        release_readiness.require_complete()

    repository_root = Path(__file__).resolve().parents[1]
    model_registry_audit = audit_model_registry(repository_root)
    model_provenance = model_provenance_dict()

    _add_v2_gate_checks(
        gate_report, input, thermo, boundary_layer, thermal, cooling,
        structural, cad_readiness, benchmark_status,
    )
    if frozen_gas is not None:
        gate_report.add(
            "physics",
            "frozen_variable_cp_q1d_closure",
            bool(performance.frozen_flow.all_closures_pass),
            value=performance.frozen_flow.as_dict()["closures"],
            limit="all energy/entropy/sonic/area/mass closures pass",
            message="Frozen-composition quasi-1-D closure failed.",
        )
        gate_report.add(
            "validation",
            "frozen_property_and_performance_benchmark",
            False,
            value="manufactured numerical regression only",
            limit=(
                "configuration-controlled CEA/property fixture and independent "
                "nozzle-performance benchmark"
            ),
            message=(
                "The variable-cp solver has conservation and constant-cp "
                "collapse tests, but no pinned physical property/performance "
                "benchmark has cleared release validation."
            ),
        )
        gate_report.add(
            "physics",
            "variable_property_boundary_layer",
            False,
            value="constant-gamma boundary-layer screen retained for diagnostics",
            limit="profile-aware viscous displacement model",
            message=(
                "Variable-cp expansion is not yet coupled to boundary-layer "
                "displacement; the reported result is diagnostic only."
            ),
        )
        gate_report.add(
            "thermal",
            "variable_property_bartz_recovery",
            False,
            value="constant-gamma Bartz/recovery screen retained for diagnostics",
            limit="profile-aware recovery enthalpy and transport properties",
            message=(
                "Variable-cp expansion is not yet coupled to Bartz recovery "
                "enthalpy/transport properties."
            ),
        )
        gate_report.add(
            "throat",
            "variable_property_discharge_coefficient",
            False,
            value="Hall correlation evaluated with chamber snapshot gamma",
            limit="validated throat-local variable-property Cd model",
            message=(
                "Hall/SP-8120 discharge coefficient remains a throat-local "
                "constant-gamma screen in variable-cp mode."
            ),
        )

    warnings.extend(contour.get("warnings", []))
    warnings.extend(gate_report.warnings)
    warnings.extend(cooling.get("warnings", []))
    if release_readiness.blocked:
        warnings.append(
            "Physical hardware release is blocked by "
            f"{len(release_readiness.blockers)} missing, invalid, or failed "
            "external evidence requirements; see report_sections."
        )
    else:
        warnings.append(
            "The configured physical-release evidence set is complete, but "
            "hardware qualification still requires the external engineering authority."
        )
    if not model_registry_audit["passed"]:
        warnings.append(
            "The physical-model provenance registry audit has unresolved entries; "
            "see report_sections.model_registry_audit."
        )
    thrust_closure.update({
        "eta_cstar": performance.eta_cstar,
        "eta_CF": performance.eta_CF,
        "effective_cstar_m_s": performance.c_star_effective,
        "required_mass_flow_kg_s": performance.m_dot,
        "spray_cstar_fixed_point_enabled": bool(
            input.spray_cstar_coupling.enabled
        ),
    })
    if injector_result is not None:
        warnings.extend(
            f"Injector {gate.name}: {gate.detail}"
            for gate in injector_result.gates
            if gate.status in {"warn", "fail"}
        )
    report_sections = {
        "thermochemistry": {
            "mode": thermo.mode,
            "source": thermo.source,
            "cea_available": thermo.cea_available,
            "chamber_state": thermo.chamber_state,
            "exit_state": thermo.exit_state,
            "expansion_model": performance.expansion_model,
            "frozen_expansion": (
                performance.frozen_flow.as_dict()
                if performance.frozen_flow is not None else None
            ),
        },
        "boundary_layer": boundary_layer,
        "thrust_closure": thrust_closure,
        "chamber_geometry": {
            "L_star": contour["L_star"],
            "contraction_ratio": contour["contraction_ratio"],
            "shoulder_radius_factor": contour["shoulder_radius_factor"],
            "shoulder_radius_source": contour["chamber"].get(
                "shoulder_radius_source"
            ),
            "shoulder_fill_fraction": contour["chamber"].get(
                "shoulder_fill_fraction"
            ),
            "throat_geometry": contour["chamber"].get("throat_geometry"),
            "throat_upstream_radius_source": contour["chamber"].get(
                "throat_upstream_radius_source"
            ),
            "throat_discharge_coefficient_hall": contour["chamber"].get(
                "throat_discharge_coefficient_hall"
            ),
            "throat_cd_target": contour["chamber"].get("throat_cd_target"),
            "minimum_cylindrical_length": contour[
                "minimum_cylindrical_length"
            ],
            "injector_to_throat_length": contour["chamber"][
                "injector_to_throat_length"
            ],
            "target_volume": contour["V_target"],
            "measured_volume": contour["V_chamber"],
            "geometry_checks": contour["geometry_checks"],
        },
        "thermal": thermal,
        "cooling": cooling,
        "injector": (
            injector_result.to_dict()
            if injector_result is not None
            else {"type": "none", "status": "disabled", "feasible": True}
        ),
        "spray_cstar_coupling": (
            spray_coupling_result.to_dict()
            if spray_coupling_result is not None
            else {
                "enabled": False,
                "status": "disabled",
                "reason": (
                    "The correlation screen does not alter cycle mass flow unless "
                    "explicit eta_mixing and eta_combustion are supplied."
                ),
            }
        ),
        "injector_interface_resolution": (
            interface_resolution.to_dict()
            if interface_resolution is not None
            else {"status": "not_requested"}
        ),
        "injector_interface": injector_interface.to_dict(),
        "structural": structural,
        "hardware_mass": hardware_mass,
        "cad_readiness": cad_readiness,
        "benchmark_status": benchmark_status,
        "model_registry_audit": model_registry_audit,
        "model_provenance": model_provenance,
        "physical_release_readiness": release_readiness.to_dict(),
    }

    if input.mode == DESIGN_MODE_VALIDATED and not gate_report.passed:
        raise RuntimeError(
            "Validated design gates failed: " + "; ".join(gate_report.warnings)
        )
    if (
        injector_result is not None
        and not injector_result.feasible
        and not input.injector.allow_infeasible
    ):
        failures = "; ".join(
            f"{gate.name}: {gate.detail}"
            for gate in injector_result.gates
            if gate.status == "fail"
        )
        raise RuntimeError(
            "Pintle injector gates failed; no design/CAD artifacts were "
            f"written. Set injector.allow_infeasible=True only for explicit "
            f"preliminary diagnostics. {failures}"
        )
    if input.strict_gates and not gate_report.passed:
        raise RuntimeError(
            "Design gates failed: " + "; ".join(gate_report.warnings)
        )

    files = _write_v2_artifacts(input, contour, gate_report, report_sections,
                                injector_result=injector_result)
    return ValidatedDesignResult(
        input=input,
        thermochemistry=thermo,
        propellant=prop,
        contour=contour,
        performance=performance,
        gate_report=gate_report,
        report_sections=report_sections,
        warnings=_dedupe(warnings),
        files=files,
    )


def design_nozzle(request: NozzleDesignRequest) -> DesignResult:
    """Generate contour, performance, design gates, and optional artifacts."""
    prop, warnings = propellant_from_request(
        propellant_name=request.propellant_name,
        use_cea=request.use_cea,
        Pc=request.Pc,
        mixture_ratio=request.mixture_ratio,
        oxidizer=request.oxidizer,
        fuel=request.fuel,
        eta_Isp=request.eta_Isp,
    )

    Rt = request.Rt
    if Rt is None:
        if request.target_thrust is None:
            raise ValueError("Either Rt or target_thrust must be provided.")
        Rt = throat_radius_for_target_thrust(
            request.target_thrust, request.Pc, request.Pa,
            request.epsilon, prop,
        )
        request.Rt = Rt

    if request.method == "bezier":
        contour = bell_nozzle_contour(
            Rt, request.epsilon, request.theta_n, request.theta_e,
            request.length_pct, gamma=prop.gamma,
        )
    else:
        contour = bell_nozzle_contour(
            Rt, request.epsilon, method=request.method,
            length_pct=request.length_pct, gamma=prop.gamma,
        )

    performance = compute_engine_performance(
        request.Pc, request.Pa, Rt, request.epsilon, prop,
    )
    gate_report = evaluate_design_gates(
        contour, request.Pc, request.Pa, prop.gamma,
        wall_thickness=request.wall_thickness,
        flange_od=request.flange_od,
        flange_length=request.flange_length,
    )

    warnings.extend(contour.get("warnings", []))
    warnings.extend(gate_report.warnings)
    files = _write_artifacts(request, contour, gate_report)

    result = DesignResult(
        request=request,
        propellant=prop,
        contour=contour,
        performance=performance,
        gate_report=gate_report,
        warnings=_dedupe(warnings),
        files=files,
    )
    if request.strict_gates and not gate_report.passed:
        raise RuntimeError(
            "Design gates failed: " + "; ".join(gate_report.warnings)
        )
    return result


def throat_radius_for_target_thrust(
    target_thrust: float,
    Pc: float,
    Pa: float,
    epsilon: float,
    prop: Propellant,
    *,
    frozen_gas=None,
) -> float:
    if target_thrust <= 0.0:
        raise ValueError("target_thrust must be positive")
    if frozen_gas is not None:
        from raosim.frozen_flow import solve_frozen_nozzle_expansion

        expansion = solve_frozen_nozzle_expansion(
            frozen_gas,
            chamber_pressure_pa=Pc,
            chamber_temperature_k=prop.Tc,
            expansion_ratio=epsilon,
            ambient_pressure_pa=Pa,
        )
        cf_ideal = expansion.thrust_coefficient
    else:
        Me = mach_from_area_ratio(epsilon, prop.gamma, supersonic=True)
        pe_pc = isentropic_pressure_ratio(Me, prop.gamma)
        cf_ideal = thrust_coefficient(Me, prop.gamma, pe_pc, Pa / Pc, epsilon)
    # Thrust = Cf_actual·Pc·At depends on the NOZZLE efficiency only;
    # combustion (c*) efficiency affects mass flow, not the thrust at fixed
    # Pc/At.  (For legacy single-eta_Isp propellants eta_CF == eta_Isp.)
    cf_actual = cf_ideal * prop.eta_CF
    if cf_actual <= 0.0:
        raise ValueError("target thrust cannot be met with non-positive Cf")
    At = target_thrust / (cf_actual * Pc)
    return math.sqrt(At / math.pi)


def _propellant_with_eta_cstar(
    propellant: Propellant,
    eta_cstar: float,
) -> Propellant:
    """Clone a gas state while changing only delivered c-star efficiency.

    CEA-backed propellants may carry an ideal ``c_star`` that differs from the
    value reconstructed from their chamber snapshot.  Preserve that explicit
    value while rebuilding the efficiency split, rather than mutating the
    shared built-in propellant table or silently replacing the CEA result.
    """

    clone = Propellant(
        name=propellant.name,
        gamma=propellant.gamma,
        Mw=propellant.Mw,
        Tc=propellant.Tc,
        eta_cstar=float(eta_cstar),
        eta_CF=float(propellant.eta_CF),
        OF=propellant.OF,
        source=propellant.source,
    )
    clone.c_star = float(propellant.c_star)
    return clone


def _validate_design_input(input: DesignInput) -> None:
    if input.mode not in DESIGN_MODES:
        raise ValueError("mode must be 'preliminary' or 'validated'")
    if input.method not in {"bezier", "moc", "rao", "rao_variational_moc"}:
        raise ValueError("method must be one of: bezier, moc, rao, rao_variational_moc")
    if input.thermo.expansion_model not in {
        "constant_gamma", "frozen_variable_cp"
    }:
        raise ValueError(
            "thermo.expansion_model must be 'constant_gamma' or "
            "'frozen_variable_cp'"
        )
    if (
        input.thermo.mode == THERMO_PINNED_CHAMBER
        and input.thermo.pinned_chamber_state is None
    ):
        raise ValueError(
            "thermo.mode='pinned_chamber_state' requires "
            "thermo.pinned_chamber_state"
        )
    if (
        input.thermo.mode != THERMO_PINNED_CHAMBER
        and input.thermo.pinned_chamber_state is not None
    ):
        raise ValueError(
            "thermo.pinned_chamber_state requires "
            "thermo.mode='pinned_chamber_state'"
        )
    if input.thermo.expansion_model == "frozen_variable_cp":
        if input.method != "bezier":
            raise ValueError(
                "frozen_variable_cp expansion is currently compatible only "
                "with bezier geometry; MOC/Rao characteristics require "
                "constant gamma"
            )
        if input.thermo.frozen_gas_table is None:
            raise ValueError(
                "frozen_variable_cp expansion requires thermo.frozen_gas_table"
            )
        if input.mode == DESIGN_MODE_VALIDATED:
            raise ValueError(
                "validated mode does not yet accept frozen_variable_cp because "
                "no configuration-controlled physical property/CFD benchmark "
                "has cleared its validation gates"
            )
    elif input.thermo.frozen_gas_table is not None:
        raise ValueError(
            "thermo.frozen_gas_table requires expansion_model='frozen_variable_cp'"
        )
    if input.Pc <= 0.0:
        raise ValueError("Pc must be positive")
    ambient_pressures = (
        float(input.ambient.Pa),
        *(float(value) for value in input.ambient.pressure_schedule_pa),
    )
    if any(not math.isfinite(value) or value < 0.0 for value in ambient_pressures):
        raise ValueError("ambient pressures must be finite and nonnegative")
    if any(value >= float(input.Pc) for value in ambient_pressures):
        raise ValueError("ambient pressures must be less than Pc")
    host_rao = input.host_rao_solver
    if host_rao.n_control < 8:
        raise ValueError("host Rao n_control must be at least 8")
    if host_rao.n_kernel < 2:
        raise ValueError("host Rao n_kernel must be at least 2")
    if host_rao.max_nfev < 0:
        raise ValueError("host Rao max_nfev must be nonnegative")
    if host_rao.solver_backend not in {"jax", "numpy"}:
        raise ValueError("host Rao solver_backend must be 'jax' or 'numpy'")
    if host_rao.wall_method not in {"coupled", "legacy", "bde"}:
        raise ValueError(
            "host Rao wall_method must be 'coupled', 'legacy', or 'bde'"
        )
    if (
        host_rao.kernel_d_fraction_max is not None
        and not 0.0 < float(host_rao.kernel_d_fraction_max) <= 1.0
    ):
        raise ValueError("host Rao kernel_d_fraction_max must be in (0, 1]")
    if (
        host_rao.physics_weight is not None
        and (
            not math.isfinite(float(host_rao.physics_weight))
            or float(host_rao.physics_weight) <= 0.0
        )
    ):
        raise ValueError("host Rao physics_weight must be finite and positive")
    if input.Rt is not None and input.Rt <= 0.0:
        raise ValueError("Rt must be positive when supplied")
    if input.target_thrust is not None and input.target_thrust <= 0.0:
        raise ValueError("target_thrust must be positive when supplied")
    if input.epsilon is not None and input.epsilon <= 1.0:
        raise ValueError("epsilon must be > 1 when supplied")
    if input.L_star is not None and input.L_star <= 0.0:
        raise ValueError("L_star must be positive when supplied")
    if input.contraction_ratio is not None and input.contraction_ratio <= 1.0:
        raise ValueError("contraction_ratio must be > 1 when supplied")
    if (
        input.shoulder_radius_factor is not None
        and input.shoulder_radius_factor <= 0.0
    ):
        raise ValueError("shoulder_radius_factor must be positive when supplied")
    if not 0.0 < input.shoulder_fill_fraction < 1.0:
        raise ValueError("shoulder_fill_fraction must be in the open interval (0, 1)")
    if (
        input.throat_cd_target is not None
        and not 0.0 < input.throat_cd_target < 1.0
    ):
        raise ValueError("throat_cd_target must be in (0, 1) when supplied")
    if (
        input.minimum_cylindrical_length is not None
        and input.minimum_cylindrical_length <= 0.0
    ):
        raise ValueError(
            "minimum_cylindrical_length must be positive when supplied"
        )
    if (
        input.cooling.coolant_inlet_temperature is not None
        and input.cooling.coolant_inlet_temperature <= 0.0
    ):
        raise ValueError("coolant_inlet_temperature must be positive when supplied")
    input.throat_geometry.validate()
    if input.mode == DESIGN_MODE_VALIDATED:
        if input.method != "bezier":
            raise ValueError("validated mode only supports the benchmarked bezier path")
        missing_chamber_inputs = [
            name for name, value in (
                ("L_star", input.L_star),
                ("contraction_ratio", input.contraction_ratio),
                ("minimum_cylindrical_length", input.minimum_cylindrical_length),
            )
            if value is None
        ]
        if missing_chamber_inputs:
            raise ValueError(
                "validated mode requires explicit chamber inputs: "
                + ", ".join(missing_chamber_inputs)
            )
        if input.thermo.mode in {
            THERMO_CONSTANT_GAMMA,
            THERMO_PINNED_CHAMBER,
        }:
            raise RuntimeError(
                "validated mode requires CEA thermochemistry from an "
                "independent evaluation"
            )
    cad = input.manufacturing.cad.lower()
    if cad not in CAD_MODES_V2:
        if cad == CAD_IPT:
            raise ValueError("Native IPT generation is deferred in v2; use STEP.")
        raise ValueError("cad must be one of: none, step, stl, both")
    if cad != CAD_NONE and input.method != "bezier":
        raise ValueError("Manufacturing CAD export is blocked for experimental methods")
    if cad in {CAD_STEP, CAD_STL, CAD_BOTH}:
        if input.manufacturing.wall_thickness is None or input.manufacturing.wall_thickness <= 0.0:
            raise ValueError("CAD export requires manufacturing.wall_thickness > 0")
    if (
        (input.interface.flange_od is None)
        != (input.interface.flange_length is None)
        and input.injector.type != "pintle"
    ):
        raise ValueError("flange_od and flange_length must be supplied together")
    if input.interface.joint_separation_factor <= 0.0:
        raise ValueError("interface.joint_separation_factor must be positive")
    for name, value in (
        ("interface.bolt_count", input.interface.bolt_count),
        ("interface.bolt_circle_diameter", input.interface.bolt_circle_diameter),
        ("interface.bolt_hole_diameter", input.interface.bolt_hole_diameter),
        ("interface.bolt_diameter", input.interface.bolt_diameter),
        ("interface.bolt_allowable_stress", input.interface.bolt_allowable_stress),
        ("interface.injector_face_od", input.interface.injector_face_od),
        ("interface.injector_face_thickness", input.interface.injector_face_thickness),
        ("interface.chamber_interface_length", input.interface.chamber_interface_length),
    ):
        if value is not None and value <= 0.0:
            raise ValueError(f"{name} must be positive when supplied")
    if input.cooling.method not in {"none", "regenerative"}:
        raise ValueError("cooling.method must be 'none' or 'regenerative'")
    if float(input.cooling.fuel_film_mass_flow or 0.0) < 0.0:
        raise ValueError("fuel_film_mass_flow must be nonnegative")
    if (
        float(input.cooling.fuel_film_mass_flow or 0.0) > 0.0
        and input.cooling.method != "regenerative"
    ):
        raise ValueError(
            "fuel_film_mass_flow requires regenerative fuel cooling so the "
            "common upstream fuel stream has an explicit jacket/film split"
        )
    if input.injector.type not in {"none", "pintle"}:
        raise ValueError("injector.type must be 'none' or 'pintle'")
    input.spray_cstar_coupling.validate()
    if input.spray_cstar_coupling.enabled and input.injector.type != "pintle":
        raise ValueError(
            "spray_cstar_coupling requires injector.type='pintle'"
        )
    if (
        input.spray_cstar_coupling.enabled
        and input.cooling.method == "regenerative"
    ):
        _, fuel_name = _thermo_feed_names(input.thermo)
        if (
            not input.cooling.coolant
            or not fuel_name
            or canonical_coolant_name(input.cooling.coolant)
            != canonical_coolant_name(fuel_name)
        ):
            raise ValueError(
                "spray_cstar_coupling with regenerative cooling requires the "
                "cycle fuel as coolant; an independent coolant/bypass needs an "
                "explicit split and mixing model"
            )
    if input.require_release_evidence and input.release_evidence_manifest is None:
        raise ValueError(
            "require_release_evidence requires release_evidence_manifest"
        )
    if input.require_release_evidence and not str(input.configuration_id or "").strip():
        raise ValueError(
            "require_release_evidence requires a nonblank configuration_id"
        )
    if input.cooling.method == "regenerative":
        if not input.cooling.channel_count or input.cooling.channel_count <= 0:
            raise ValueError("regenerative cooling requires channel_count > 0")
        if not input.cooling.channel_width or input.cooling.channel_width <= 0.0:
            raise ValueError("regenerative cooling requires channel_width > 0")
        if not input.cooling.channel_height or input.cooling.channel_height <= 0.0:
            raise ValueError("regenerative cooling requires channel_height > 0")
        if (
            input.cooling.coolant_mass_flow is not None
            and input.cooling.coolant_mass_flow <= 0.0
        ):
            raise ValueError(
                "coolant_mass_flow must be positive when supplied"
            )
        if (
            not input.spray_cstar_coupling.enabled
            and (
                not input.cooling.coolant_mass_flow
                or input.cooling.coolant_mass_flow <= 0.0
            )
        ):
            raise ValueError("regenerative cooling requires coolant_mass_flow > 0")
        if float(input.cooling.fuel_film_mass_flow or 0.0) > 0.0:
            _, fuel_name = _thermo_feed_names(input.thermo)
            if (
                not input.cooling.coolant
                or not fuel_name
                or canonical_coolant_name(input.cooling.coolant)
                != canonical_coolant_name(fuel_name)
            ):
                raise ValueError(
                    "fuel_film_mass_flow requires the cycle fuel to be the "
                    "regenerative coolant"
                )


def _thermo_feed_names(thermo: ThermoSpec) -> tuple[str | None, str | None]:
    """Return ``(oxidizer, fuel)`` for injector feed-property resolution."""
    oxidizer, fuel = thermo.oxidizer, thermo.fuel
    if (
        (not oxidizer or not fuel)
        and thermo.propellant_name
        and "/" in thermo.propellant_name
    ):
        pair_oxidizer, pair_fuel = (
            part.strip() for part in thermo.propellant_name.split("/", 1)
        )
        oxidizer = oxidizer or pair_oxidizer
        fuel = fuel or pair_fuel
    return oxidizer, fuel


def _cooling_with_split_pressure_boundary(
    input: DesignInput,
) -> tuple[CoolingSpec, list[str]]:
    """Return a cooling spec whose regen boundary follows the split dP model."""
    warnings: list[str] = []
    cooling = input.cooling
    if cooling.coolant_outlet_pressure is not None:
        if float(cooling.injector_pressure_drop or 0.0) != 0.0:
            warnings.append(
                "CoolingSpec.injector_pressure_drop is deprecated and ignored "
                "because coolant_outlet_pressure is an explicit absolute "
                "jacket outlet boundary."
            )
        return replace(cooling, injector_pressure_drop=0.0), warnings

    _, fuel_name = _thermo_feed_names(input.thermo)
    fuel_is_coolant = bool(
        cooling.method == "regenerative"
        and cooling.coolant
        and fuel_name
        and canonical_coolant_name(cooling.coolant)
        == canonical_coolant_name(fuel_name)
    )
    if fuel_is_coolant:
        if float(cooling.injector_pressure_drop or 0.0) != 0.0:
            warnings.append(
                "CoolingSpec.injector_pressure_drop is deprecated and ignored; "
                "the fuel-side regen boundary is derived from "
                "injector.fuel_dp_fraction."
            )
        return replace(
            cooling,
            injector_pressure_drop=(
                float(input.injector.fuel_dp_fraction) * float(input.Pc)
            ),
        ), warnings

    if float(cooling.injector_pressure_drop or 0.0) != 0.0:
        warnings.append(
            "CoolingSpec.injector_pressure_drop is deprecated and ignored; "
            "set coolant_outlet_pressure for an absolute regen boundary or "
            "set injector.fuel_dp_fraction for the fuel-side split dP model."
        )
    return replace(cooling, injector_pressure_drop=0.0), warnings


def _cooling_at_cycle_mass_flow(
    input: DesignInput,
    *,
    total_mass_flow: float,
    mixture_ratio: float,
) -> tuple[CoolingSpec, list[str]]:
    """Resolve cooling pressure and direct-fuel flow for one cycle iterate."""

    cooling, warnings = _cooling_with_split_pressure_boundary(input)
    if (
        input.spray_cstar_coupling.enabled
        and cooling.method == "regenerative"
    ):
        fuel_mass_flow = float(total_mass_flow) / (1.0 + float(mixture_ratio))
        film_mass_flow = float(cooling.fuel_film_mass_flow or 0.0)
        jacket_mass_flow = fuel_mass_flow - film_mass_flow
        if jacket_mass_flow <= 0.0:
            raise ValueError(
                "spray/c-star/regen closure requires cycle fuel mass flow to "
                "exceed fuel_film_mass_flow"
            )
        supplied = input.cooling.coolant_mass_flow
        if supplied is not None and not math.isclose(
            float(supplied), jacket_mass_flow, rel_tol=1.0e-9, abs_tol=0.0
        ):
            warnings.append(
                "spray/c-star/regen outer closure derives coolant_mass_flow "
                "from the current cycle fuel stream after the explicit film "
                f"bypass ({jacket_mass_flow:.9g} kg/s jacket + "
                f"{film_mass_flow:.9g} kg/s film = "
                f"{fuel_mass_flow:.9g} kg/s total fuel); "
                f"the static input value {float(supplied):.9g} kg/s is only an "
                "initial request and is not reused across iterations."
            )
        cooling = replace(cooling, coolant_mass_flow=jacket_mass_flow)
    return cooling, warnings


def _build_v2_contour(
    input: DesignInput,
    Rt: float,
    epsilon: float,
    prop: Propellant,
) -> dict:
    throat_geometry = input.throat_geometry
    throat_upstream_radius_source = "input_throat_geometry"
    if input.throat_cd_target is not None:
        radius_bounds = (
            REPOSITORY_UPSTREAM_RADIUS_RATIO_EXTENSION_BOUNDS
            if input.allow_throat_radius_extension
            else SP8120_UPSTREAM_RADIUS_RATIO_BOUNDS
        )
        throat_geometry = replace(
            throat_geometry,
            upstream_radius_ratio=upstream_radius_ratio_for_discharge_coefficient(
                input.throat_cd_target,
                prop.gamma,
                min_ratio=radius_bounds[0],
                max_ratio=radius_bounds[1],
            ),
        )
        input.throat_geometry = throat_geometry
        throat_upstream_radius_source = (
            "cd_target_hall_repository_extension"
            if throat_geometry.upstream_radius_ratio
            > SP8120_UPSTREAM_RADIUS_RATIO_BOUNDS[1]
            else "cd_target_hall_sp8120"
        )

    if input.method == "bezier":
        nozzle = bell_nozzle_contour(
            Rt, epsilon, input.theta_n, input.theta_e, input.length_pct,
            gamma=prop.gamma,
            throat_geometry=throat_geometry,
        )
    else:
        host_rao = input.host_rao_solver
        nozzle = bell_nozzle_contour(
            Rt, epsilon, method=input.method,
            length_pct=input.length_pct, gamma=prop.gamma,
            pa_over_p0=(
                float(input.ambient.design_pressure) / float(input.Pc)
            ),
            starting_line_method=host_rao.starting_line_method,
            rao_moc_n_control=host_rao.n_control,
            rao_moc_n_kernel=host_rao.n_kernel,
            rao_moc_max_nfev=host_rao.max_nfev,
            rao_moc_evaluate_moc=host_rao.evaluate_moc,
            rao_moc_theta_n_guess_deg=host_rao.theta_n_guess_deg,
            rao_moc_solver_backend=host_rao.solver_backend,
            rao_moc_wall_method=host_rao.wall_method,
            rao_moc_kernel_d_fraction_max=(
                host_rao.kernel_d_fraction_max
            ),
            rao_moc_physics_weight=host_rao.physics_weight,
            throat_geometry=throat_geometry,
        )

    contraction_ratio = float(
        input.contraction_ratio
        if input.contraction_ratio is not None else 2.5
    )
    shoulder_radius_source = "user_supplied"
    if input.shoulder_radius_factor is None:
        input.shoulder_radius_factor = auto_shoulder_factor(
            Rt,
            contraction_ratio,
            throat_geometry=throat_geometry,
            fill_fraction=input.shoulder_fill_fraction,
        )
        shoulder_radius_source = "auto_geometric_closure"
    shoulder_radius_factor = float(input.shoulder_radius_factor)
    chamber = chamber_contour(
        Rt,
        L_star=float(input.L_star if input.L_star is not None else 1.0),
        contraction_ratio=contraction_ratio,
        shoulder_radius_factor=shoulder_radius_factor,
        minimum_cylindrical_length=float(
            input.minimum_cylindrical_length
            if input.minimum_cylindrical_length is not None else 1e-6
        ),
        throat_geometry=throat_geometry,
    )
    chamber["shoulder_radius_source"] = shoulder_radius_source
    chamber["shoulder_fill_fraction"] = float(input.shoulder_fill_fraction)
    chamber["throat_upstream_radius_source"] = throat_upstream_radius_source
    chamber["throat_discharge_coefficient_hall"] = (
        throat_discharge_coefficient_hall(
            throat_geometry.upstream_radius_ratio, prop.gamma
        )
    )
    chamber["throat_cd_target"] = input.throat_cd_target
    chamber["allow_throat_radius_extension"] = (
        input.allow_throat_radius_extension
    )
    contour = full_engine_contour(chamber, nozzle)
    contour["geometry_checks"] = thrust_chamber_geometry_checks(
        contour,
        offset_distance=input.manufacturing.wall_thickness,
    )
    return contour


def _resolved_max(current, resolved):
    """Use the resolved layout value unless an explicit larger value exists."""
    if current is None:
        return resolved
    return max(float(current), float(resolved))


def _apply_pintle_interface_resolution(
    input: DesignInput,
    contour: dict,
    injector_result: Any | None = None,
):
    """Synchronize the chamber flange and pintle face bolt pattern.

    The first call happens before pintle sizing so injector gates have a real
    face OD and bolt pattern.  A second call after pintle sizing can fold in
    the machined faceplate's manifold-driven minimums, keeping the chamber
    flange STEP and injector STEP on the same bolt-together interface.
    """

    if input.injector.type != "pintle":
        return None

    chamber = contour.get("chamber", {})
    chamber_radius = float(chamber.get("Rc", contour["y"][0]))
    min_feature = getattr(input.injector.manufacturing, "min_feature", None)
    min_tool = getattr(input.injector.mechanical, "min_tool_diameter", None)
    min_face_od = min_face_t = min_bcd = min_hole = None

    if injector_result is not None:
        try:
            from raosim.injector_cad import resolve_machined_pintle_layout

            layout = resolve_machined_pintle_layout(
                injector_result, spec=input.injector
            )
            resolved = layout["resolved"]
            min_face_od = resolved["faceplate_outer_diameter_m"]
            min_face_t = resolved["faceplate_thickness_m"]
            min_bcd = resolved["bolt_circle_diameter_m"]
            min_hole = resolved["bolt_hole_diameter_m"]
        except Exception:
            # The generic bolted-interface resolver is still useful without
            # the machined layout report; artifact export captures CAD issues.
            pass

    resolution = resolve_bolted_interface_geometry(
        chamber_pressure=input.Pc,
        chamber_radius=chamber_radius,
        wall_thickness=input.manufacturing.wall_thickness,
        flange_outer_diameter=input.interface.flange_od,
        flange_length=input.interface.flange_length,
        face_outer_diameter=input.interface.injector_face_od,
        face_thickness=input.interface.injector_face_thickness,
        bolt_count=input.interface.bolt_count,
        bolt_circle_diameter=input.interface.bolt_circle_diameter,
        bolt_hole_diameter=input.interface.bolt_hole_diameter,
        bolt_diameter=input.interface.bolt_diameter,
        bolt_allowable_stress=input.interface.bolt_allowable_stress,
        material_yield_strength=input.material.yield_strength,
        structural_fos=float(input.material.structural_fos),
        min_feature=min_feature,
        min_tool_diameter=min_tool,
        minimum_face_outer_diameter=min_face_od,
        minimum_face_thickness=min_face_t,
        minimum_bolt_circle_diameter=min_bcd,
        minimum_bolt_hole_diameter=min_hole,
        joint_separation_factor=input.interface.joint_separation_factor,
    )

    input.interface.flange_od = _resolved_max(
        input.interface.flange_od, resolution.flange_outer_diameter
    )
    input.interface.flange_length = _resolved_max(
        input.interface.flange_length, resolution.flange_length
    )
    input.interface.injector_face_od = _resolved_max(
        input.interface.injector_face_od, resolution.face_outer_diameter
    )
    input.interface.injector_face_thickness = _resolved_max(
        input.interface.injector_face_thickness, resolution.face_thickness
    )
    input.interface.bolt_count = max(
        int(input.interface.bolt_count or 0), int(resolution.bolt_count)
    )
    input.interface.bolt_circle_diameter = _resolved_max(
        input.interface.bolt_circle_diameter, resolution.bolt_circle_diameter
    )
    input.interface.bolt_hole_diameter = _resolved_max(
        input.interface.bolt_hole_diameter, resolution.bolt_hole_diameter
    )

    geo = input.injector.geometry
    mech = input.injector.mechanical
    geo.face_od = _resolved_max(geo.face_od, input.interface.injector_face_od)
    geo.face_thickness = _resolved_max(
        geo.face_thickness, input.interface.injector_face_thickness
    )
    mech.bolt_count = max(int(mech.bolt_count or 0), int(input.interface.bolt_count))
    mech.bolt_circle_diameter = _resolved_max(
        mech.bolt_circle_diameter, input.interface.bolt_circle_diameter
    )
    mech.bolt_hole_diameter = _resolved_max(
        mech.bolt_hole_diameter, input.interface.bolt_hole_diameter
    )
    mech.faceplate_outer_diameter = _resolved_max(
        mech.faceplate_outer_diameter, input.interface.injector_face_od
    )
    mech.faceplate_thickness = _resolved_max(
        mech.faceplate_thickness, input.interface.injector_face_thickness
    )
    return resolution


def _add_v2_gate_checks(
    report: DesignGateReport,
    input: DesignInput,
    thermo: ThermochemistryResult,
    boundary_layer: dict,
    thermal: dict,
    cooling: dict,
    structural: dict,
    cad_readiness: dict,
    benchmark_status: dict,
) -> None:
    report.add(
        "workflow", "official_preliminary_path",
        input.method == "bezier",
        value=input.method, limit="bezier",
        message="Only the Bezier Rao/TOP path is eligible for validated design outputs.",
    )
    report.add(
        "workflow", "validated_thermochemistry",
        input.mode != DESIGN_MODE_VALIDATED or thermo.source == "rocketcea",
        value=thermo.source, limit="rocketcea in validated mode",
        message="Validated mode requires RocketCEA-derived thermochemistry.",
    )
    report.add(
        "workflow", "benchmark_status",
        bool(benchmark_status["validated_for_design"]),
        value=benchmark_status["status"], limit="benchmarked",
        message="Experimental MOC/Rao methods remain blocked until literature benchmarks pass.",
    )
    report.add(
        "physics", "boundary_layer_area_loss",
        float(boundary_layer["epsilon_loss_fraction"]) <= 0.08,
        value=float(boundary_layer["epsilon_loss_fraction"]), limit="<= 0.08",
        message="Boundary-layer displacement causes excessive effective area-ratio loss.",
    )
    report.add(
        "thermal", "heat_flux_limit",
        float(structural["heat_flux_margin"]) >= 1.0,
        value=float(thermal["q_max"]), limit=f"<= {input.material.max_heat_flux}",
        message="Estimated heat flux exceeds the material screening limit.",
    )
    report.add(
        "thermal", "wall_temperature_limit",
        float(structural["temperature_margin"]) >= 1.0,
        value=float(structural["wall_temperature"]), limit=f"<= {input.material.max_temperature}",
        message="Estimated wall temperature exceeds the material limit.",
    )
    report.add(
        "cooling", "regenerative_cooling_required_for_validated",
        input.mode != DESIGN_MODE_VALIDATED or input.cooling.method == "regenerative",
        value=input.cooling.method, limit="regenerative in validated mode",
        message="Validated hardware-oriented runs require a regenerative cooling screen.",
    )
    report.add(
        "cooling", "cooling_margin",
        input.cooling.method != "regenerative" or float(cooling["cooling_margin"]) >= 1.0,
        value=float(cooling["cooling_margin"]), limit=">= 1",
        message="Regenerative cooling screen has insufficient thermal margin.",
    )
    report.add(
        "structural", "hoop_stress_margin",
        float(structural["stress_margin"]) >= float(input.material.structural_fos),
        value=float(structural["stress_margin"]),
        limit=f">= {float(input.material.structural_fos):g}",
        message="Thin-wall hoop stress screening margin is too low.",
    )
    report.add(
        "cad", "step_authoritative",
        bool(cad_readiness["step_authoritative"]),
        value=cad_readiness["requested_cad"], limit="STEP/STL with STEP authoritative",
        message="STEP remains the authoritative CAD artifact; native IPT is deferred.",
    )


def _hardware_mass_section(
    input: DesignInput,
    contour: dict,
    cooling_spec: CoolingSpec,
    cooling: dict,
    interface_resolution: Any,
    injector_result: Any,
) -> dict:
    """Build the geometry-integrated hardware mass ledger for the report.

    The thrust-chamber entry integrates the same ``RegenWallProfile`` that
    :mod:`raosim.regen_cad` revolves; the injector entry integrates the same
    machined layout that :mod:`raosim.injector_cad` cuts.  Anything that cannot
    be resolved is reported as unavailable with a reason, never as zero -- see
    :mod:`raosim.mass_ledger`.
    """

    from raosim.mass_ledger import (
        MassLedger,
        combine_ledgers,
        flange_bolt_mass_ledger,
        injector_mass_ledger,
        thrust_chamber_mass_ledger,
    )
    from raosim.regen_profile import RegenWallProfile

    ledgers: list[MassLedger] = []
    notes: list[str] = []

    material = input.material
    if getattr(material, "density", None) is None:
        try:
            material = MaterialSpec.from_catalog(material.name)
        except Exception:
            notes.append(
                f"material '{getattr(material, 'name', material)}' has no "
                "density and is not in the raosim.materials catalog; hardware "
                "mass cannot be computed"
            )

    # ---- thrust chamber -------------------------------------------------- #
    t_hot = input.manufacturing.wall_thickness
    n_ch = getattr(cooling_spec, "channel_count", None)
    w_ch = getattr(cooling_spec, "channel_width", None)
    h_ch = getattr(cooling_spec, "channel_height", None)
    regenerative = str(getattr(cooling_spec, "method", "none")) == "regenerative"
    if regenerative and all(
        v is not None and float(v) > 0.0 for v in (t_hot, n_ch, w_ch, h_ch)
    ):
        # Reuse the solved land-width profile when the coupled cooling analysis
        # produced one, so the mass integral and the fin/heat-transfer model
        # describe the same rib.
        land = cooling.get("land_width")
        t_jacket, jacket_note, jacket_material = _closeout_thickness(
            contour, cooling, input, t_hot=float(t_hot),
            channel_height=float(h_ch),
        )
        notes.append(jacket_note)
        profile = RegenWallProfile.uniform(
            contour,
            channel_count=int(n_ch),
            channel_width=float(w_ch),
            channel_height=float(h_ch),
            t_hot=float(t_hot),
            land_width=land,
            t_jacket=t_jacket,
            helix_turns=float(cooling.get("helix_turns", 0.0) or 0.0),
        )
        ledgers.append(thrust_chamber_mass_ledger(
            profile,
            liner_material=material,
            closeout_material=jacket_material,
            joint_allowance=_joint_allowance(input.manufacturing),
        ))
    else:
        notes.append(
            "thrust-chamber mass needs a regenerative wall with a positive "
            "wall thickness and channel count/width/height"
        )

    # ---- bolted chamber/injector interface ------------------------------- #
    # The flange and fasteners are structure, not a hot wall, so they are
    # priced in the jacket/structure alloy rather than the copper liner --
    # SP-8087 sec. 2.1.3.1 on jacket and reinforcement materials.
    structural_material = None
    try:
        structural_material = MaterialSpec.from_catalog(
            getattr(input.material, "jacket_material", None) or "Inconel 718"
        )
    except Exception:
        structural_material = material
    if interface_resolution is not None:
        ledgers.append(flange_bolt_mass_ledger(
            interface_resolution, flange_material=structural_material,
        ))
    else:
        notes.append(
            "no bolted chamber/injector interface was resolved, so flange and "
            "fastener mass is absent from this ledger"
        )

    # ---- injector -------------------------------------------------------- #
    if injector_result is not None and input.injector.type == "pintle":
        try:
            from raosim.injector_cad import resolve_machined_pintle_layout

            layout = resolve_machined_pintle_layout(
                injector_result, spec=input.injector
            )
            ledgers.append(injector_mass_ledger(
                layout, body_material=structural_material,
            ))
        except Exception as exc:  # layout screens raise on infeasible geometry
            notes.append(
                f"injector hardware mass unavailable: the machined pintle "
                f"layout could not be resolved ({exc})"
            )
    else:
        notes.append(
            "injector hardware mass is only modelled for the pintle injector"
        )

    if not ledgers:
        return {
            "status": "unavailable",
            "complete": False,
            "total_mass_kg": None,
            "unavailable_reason": "; ".join(notes),
            "notes": notes,
        }

    combined = combine_ledgers(ledgers, scope="engine_hardware")
    out = combined.to_dict()
    out["status"] = "resolved" if combined.complete else "partial"
    out["notes"] = notes
    # The bolted joint is the single largest mass item in this ledger, and its
    # layout defaults are spacing heuristics rather than load paths.  Report
    # what a mass-minimising fastener selection would give, so the gap is
    # visible without silently changing the resolved geometry.
    out["joint_sizing_opportunity"] = _joint_sizing_opportunity(
        input, contour, structural_material
    )
    out["excludes"] = [
        "propellant valves, lines, manifolds and their brackets",
        "gimbal, thrust take-out structure and engine mounts",
        "igniter hardware, seals, gaskets and instrumentation",
        "the electric feed system (see the electric-pump hardware BOM)",
    ]
    return out


# Fallback closeout thickness as a multiple of the hot-gas wall, used only when
# the coolant pressure profile is unavailable.  Matches
# ``MissionSpec.closeout_thickness_ratio``.
_CLOSEOUT_THICKNESS_RATIO = 2.0
# Manufacturing floor and thin-shell limit, matching the MDO MissionSpec
# defaults so both pipelines size the same jacket.
_CLOSEOUT_THICKNESS_MIN = 5.0e-4
_CLOSEOUT_THIN_SHELL_RATIO_MAX = 1.0 / 15.0


def _closeout_thickness(
    contour: dict,
    cooling: dict,
    input: DesignInput,
    *,
    t_hot: float,
    channel_height: float,
) -> tuple[Any, str, Any]:
    """Structural jacket thickness from the SP-125 outer-shell hoop screen.

    NASA SP-125, printed p. 109, on the coaxial-shell chamber: *"the outer
    shell is subjected only to the hoop stress induced by the coolant
    pressure"*, so per station

        t_j = FoS * p_coolant * r_outer / sigma_yield

    floored at a manufacturing minimum.  A tapered jacket is normal practice --
    NASA SP-8087 sec. 2.1.3.1: *"The brazed jacket can be tapered for optimum
    strength and weight"* -- so the thickness follows the local pressure and
    radius.  SP-8087 sec. 2.1.3 gives the factors of safety in use (yield
    1.0-1.32, ultimate 1.3-1.8); the conservative end of the yield band is
    taken.

    Returns ``(thickness, note, jacket_material)``.  ``thickness`` is a
    per-station array when the coolant pressure profile is available and a
    scalar fallback otherwise, so the caller never silently loses the
    distinction.
    """

    jacket_name = getattr(input.material, "jacket_material", None) or "Inconel 718"
    try:
        jacket = MaterialSpec.from_catalog(jacket_name)
    except Exception:
        return (
            t_hot * _CLOSEOUT_THICKNESS_RATIO,
            (
                f"jacket material '{jacket_name}' is not in the raosim.materials "
                f"catalog; the closeout fell back to "
                f"{_CLOSEOUT_THICKNESS_RATIO:g} x the hot-gas wall, which is an "
                "assumption rather than a hoop-sized jacket"
            ),
            None,
        )

    p_cool = cooling.get("coolant_pressure")
    radius = np.asarray(contour["y"], dtype=float)
    r_outer = radius + t_hot + channel_height
    p_arr = np.asarray(p_cool, dtype=float) if p_cool is not None else None
    if p_arr is None or p_arr.ndim == 0 or p_arr.shape != radius.shape:
        # A scalar jacket pressure still supports the hoop screen; only the
        # taper is lost.
        p_scalar = (
            float(p_arr) if p_arr is not None and p_arr.ndim == 0
            else _finite_positive_or_none(cooling.get("coolant_outlet_pressure"))
        )
        if p_scalar is None:
            return (
                t_hot * _CLOSEOUT_THICKNESS_RATIO,
                (
                    "no jacket pressure was solved, so the closeout fell back "
                    f"to {_CLOSEOUT_THICKNESS_RATIO:g} x the hot-gas wall; that "
                    "is an assumption, not an SP-125 hoop-sized jacket"
                ),
                jacket,
            )
        p_arr = np.full_like(radius, p_scalar)
        taper = "uniform (only a scalar jacket pressure was available)"
    else:
        taper = "tapered per station (SP-8087 sec. 2.1.3.1)"

    fos = float(getattr(input.material, "closeout_structural_fos", 1.32))
    sigma = float(jacket.yield_strength)
    t_req = fos * p_arr * r_outer / max(sigma, 1.0)
    t_j = np.maximum(t_req, _CLOSEOUT_THICKNESS_MIN)
    ratio_max = float(np.max(t_j / np.maximum(r_outer, 1e-12)))
    note = (
        f"structural closeout sized by the SP-125 p.109 outer-shell hoop "
        f"screen t = {fos:g} x p_coolant x r_outer / {sigma/1e6:.0f} MPa "
        f"({jacket_name}), {taper}, floored at "
        f"{_CLOSEOUT_THICKNESS_MIN*1e3:g} mm: "
        f"{float(np.min(t_j))*1e3:.3f}-{float(np.max(t_j))*1e3:.3f} mm"
    )
    if ratio_max > _CLOSEOUT_THIN_SHELL_RATIO_MAX:
        note += (
            f"; WARNING max t/r = {ratio_max:.4f} exceeds SP-125's thin-shell "
            f"validity limit of {_CLOSEOUT_THIN_SHELL_RATIO_MAX:.4f} (printed "
            "p. 336), so the hoop formula is outside its own model -- the "
            "jacket alloy or the jacket pressure needs to change"
        )
    return t_j, note, jacket


def _joint_sizing_opportunity(
    input: DesignInput, contour: dict, structural_material: Any,
) -> dict:
    """What a mass-minimising bolted joint would give, versus the layout default.

    The resolved interface uses spacing heuristics -- a bolt hole at
    ``0.06 x chamber diameter``, a bolt circle at ``chamber OD + 6 x hole``, a
    faceplate at ``2 x hole``.  Those are not load paths, and on the 13 kN
    baseline they made the flange and faceplate about three quarters of the
    engine's modelled hardware mass.  This reports the alternative rather than
    imposing it, because changing the resolved joint changes exported CAD.
    """

    from raosim.interface import size_bolted_interface

    density = getattr(structural_material, "density", None)
    yield_strength = getattr(structural_material, "yield_strength", None)
    try:
        sizing = size_bolted_interface(
            chamber_radius=float(contour["chamber"]["Rc"]),
            chamber_pressure=float(input.Pc),
            wall_thickness=input.manufacturing.wall_thickness,
            material_yield_strength=yield_strength,
            structural_fos=float(
                getattr(structural_material, "structural_fos", 1.5)
            ),
            flange_density=density,
            bolt_density=density,
        )
    except Exception as exc:
        return {
            "status": "unavailable",
            "reason": f"bolted-joint sizing did not converge: {exc}",
        }
    payload = sizing.to_dict()
    payload["status"] = "advisory"
    payload["applied_to_exported_geometry"] = False
    payload["note"] = (
        "advisory only -- the exported flange, bolt pattern and faceplate "
        "still come from resolve_bolted_interface_geometry.  Pass the selected "
        "bolt count, hole diameter and faceplate thickness through "
        "InterfaceSpec to adopt this joint."
    )
    return payload


def _finite_positive_or_none(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0.0 else None


def _joint_allowance(manufacturing: ManufacturingSpec) -> float:
    """Weld/braze mass allowance, after the SP-125 eq. 5-16 weld-land idea.

    Returns 1.0 (no allowance) unless the caller populated
    ``ManufacturingSpec.weld_allowance`` / ``braze_allowance``.  Those fields
    are fractional build-ups, so 0.05 means +5% metal.
    """

    total = 0.0
    for value in (
        getattr(manufacturing, "weld_allowance", None),
        getattr(manufacturing, "braze_allowance", None),
    ):
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v) and v > 0.0:
            total += v
    return 1.0 + total


def _cad_readiness(
    input: DesignInput,
    contour: dict,
    gate_report: DesignGateReport,
) -> dict:
    cad = input.manufacturing.cad.lower()
    max_od = 2.0 * float(max(contour["y"]))
    chamber_section = contour.get("chamber", {})
    interface_od = 2.0 * float(chamber_section.get("Rc", max(contour["y"])))
    flange_ok = (
        input.interface.flange_od is None
        or (
            input.interface.flange_length is not None
            and input.interface.flange_od > interface_od
            and input.interface.flange_length > 0.0
        )
    )
    placeholders = {
        "throat_insert": bool(input.manufacturing.throat_insert),
        "chamber_nozzle_interface": input.interface.chamber_interface_length is not None,
        "flange_pattern": (
            input.interface.bolt_count is not None
            and input.interface.bolt_circle_diameter is not None
            and input.interface.bolt_hole_diameter is not None
        ),
        "tolerances": input.manufacturing.tolerance is not None,
        "weld_braze_allowances": (
            input.manufacturing.weld_allowance is not None
            or input.manufacturing.braze_allowance is not None
        ),
    }
    return {
        "requested_cad": cad,
        "step_authoritative": cad in {CAD_NONE, CAD_STEP, CAD_BOTH},
        "native_ipt_deferred": True,
        "manufacturing_step_allowed": (
            input.method == "bezier"
            and cad in {CAD_STEP, CAD_BOTH}
            and input.manufacturing.wall_thickness is not None
            and flange_ok
        ),
        "flange_ok": flange_ok,
        "max_nozzle_od": max_od,
        "interface_reference_od": interface_od,
        "placeholders": placeholders,
        "legacy_gate_passed_before_v2": gate_report.passed,
    }


def _injector_interface_screen(input: DesignInput, contour: dict):
    geometry = getattr(input.injector, "geometry", None)
    face_od = input.interface.injector_face_od
    if face_od is None and geometry is not None:
        face_od = getattr(geometry, "face_od", None)
    face_thickness = input.interface.injector_face_thickness
    if face_thickness is None and geometry is not None:
        face_thickness = getattr(geometry, "face_thickness", None)

    chamber = contour.get("chamber", {})
    chamber_radius = float(chamber.get("Rc", contour["y"][0]))
    return screen_injector_chamber_interface(
        chamber_pressure=input.Pc,
        chamber_radius=chamber_radius,
        wall_thickness=input.manufacturing.wall_thickness,
        face_outer_diameter=face_od,
        face_thickness=face_thickness,
        flange_outer_diameter=input.interface.flange_od,
        flange_length=input.interface.flange_length,
        bolt_count=input.interface.bolt_count,
        bolt_circle_diameter=input.interface.bolt_circle_diameter,
        bolt_hole_diameter=input.interface.bolt_hole_diameter,
        bolt_diameter=input.interface.bolt_diameter,
        bolt_allowable_stress=input.interface.bolt_allowable_stress,
        material_yield_strength=input.material.yield_strength,
        material_elastic_modulus=input.material.elastic_modulus,
        material_poisson_ratio=input.material.poisson_ratio,
        joint_separation_factor=input.interface.joint_separation_factor,
    )


def _benchmark_status(method: str, contour: dict | None = None) -> dict:
    if method == "bezier":
        if contour is not None and contour.get("rao_chart_extrapolated", False):
            return {
                "status": "unvalidated_rao_chart_extrapolation",
                "validated_for_design": False,
                "notes": [
                    "Bezier angles were extrapolated outside the digitized "
                    "Rao/TOP chart domain; supply in-domain inputs or explicit "
                    "angles backed by a separate benchmark."
                ],
            }
        return {
            "status": "benchmarked_preliminary_top_geometry",
            "validated_for_design": True,
            "notes": ["Bezier/TOP path is the trusted preliminary baseline."],
        }
    return {
        "status": "experimental_with_strict_literature_benchmark",
        "validated_for_design": False,
        "notes": [
            "The Rao 1958 Nozzle-B case is a strict published-data benchmark, "
            "but generic MOC/Rao contours remain diagnostic and blocked from "
            "manufacturing outputs until their per-run residual, topology, "
            "valid-region, thrust, and source-configuration gates pass."
        ],
    }


def _write_artifacts(
    request: NozzleDesignRequest,
    contour: dict,
    gate_report: DesignGateReport,
) -> dict[str, Path]:
    if request.output_dir is None:
        return {}

    out = Path(request.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    files["csv"] = export_csv(
        contour["x"], contour["y"], out / "nozzle_profile.csv",
        request.csv_points,
    )

    metadata = {
        "design_status": contour.get("design_status"),
        "hardware_qualified": False,
        "gate_passed": gate_report.passed,
    }

    cad = request.cad.lower()
    if cad in {"step", "both", "ipt"}:
        step_path = export_step(
            contour["x"], contour["y"], out / "nozzle.step",
            request.angular_points,
            wall_thickness=request.wall_thickness,
            flange_od=request.flange_od,
            flange_length=request.flange_length,
            metadata=metadata,
        )
        files["step"] = step_path

    if cad == "both":
        files["stl"] = export_stl(
            contour["x"], contour["y"], out / "nozzle.stl",
            request.angular_points,
            wall_thickness=request.wall_thickness,
            flange_od=request.flange_od,
            flange_length=request.flange_length,
        )

    if cad in {"ipt", "both"}:
        if "step" not in files:
            raise ValueError("IPT packaging requires a STEP artifact.")
        files["ipt_manifest"] = package_ipt_request(
            files["step"], out / "nozzle_ipt_manifest.json", metadata=metadata,
        )

    report_path = out / "design_report.json"
    report_path.write_text(
        json.dumps(
            _json_ready(gate_report.to_dict()),
            indent=2,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    files["design_report"] = report_path
    return files


def _write_v2_artifacts(
    input: DesignInput,
    contour: dict,
    gate_report: DesignGateReport,
    report_sections: dict[str, Any],
    injector_result: Any | None = None,
) -> dict[str, Path]:
    if input.manufacturing.output_dir is None:
        return {}

    out = Path(input.manufacturing.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    files["csv"] = export_csv(
        contour["x"], contour["y"], out / "thrust_chamber_profile.csv",
        input.manufacturing.csv_points,
    )

    # Pintle deliverable folder: labeled schematic (SVG+PNG), parameters JSON,
    # dimensions CSV (always) + optional STEP/STL/DXF reference geometry.
    # Diagnostic-only output (cad=none) remains best effort.  If CAD was
    # explicitly requested, a failed B-rep/topology/export gate is fatal; an
    # EXPORT_ERROR marker is still written so the failure is inspectable.
    if injector_result is not None and input.injector.type == "pintle":
        pintle_dir = out / "pintle"
        injector_cad_mode = str(
            getattr(input.injector, "cad", "none") or "none"
        ).strip().lower()
        try:
            from raosim.injector_export import export_pintle_package
            pkg = export_pintle_package(
                injector_result, pintle_dir, spec=input.injector,
                cad=getattr(input.injector, "cad", "none"),
                cad_format=getattr(input.injector, "cad_format", "step"))
            files["pintle_dir"] = Path(pkg["dir"])
            for key, path in pkg["files"].items():
                files[f"pintle_{key}"] = Path(path)
        except Exception as exc:           # pragma: no cover - defensive
            import traceback
            pintle_dir.mkdir(parents=True, exist_ok=True)
            (pintle_dir / "EXPORT_ERROR.txt").write_text(
                "pintle package export failed:\n" + traceback.format_exc(),
                encoding="utf-8")
            files["pintle_error"] = pintle_dir / "EXPORT_ERROR.txt"
            if injector_cad_mode not in {"none", "off"}:
                raise RuntimeError(
                    "Requested pintle CAD failed a required geometry/export "
                    f"gate; see {files['pintle_error']}"
                ) from exc

    metadata = _v2_metadata(input, contour, gate_report, report_sections)
    seal_center = seal_width = None
    if injector_result is not None and input.injector.type == "pintle":
        try:
            from raosim.injector_cad import resolve_machined_pintle_layout

            interface_layout = resolve_machined_pintle_layout(
                injector_result, spec=input.injector
            )["resolved"]
            if interface_layout["seal_type"] == "o_ring":
                seal_center = interface_layout["seal_center_radius_m"]
                seal_width = interface_layout["o_ring_groove_width_m"]
        except Exception:
            # The common bolt pattern is still exported.  Missing optional
            # seal metadata is captured in the design report/CAD readiness
            # rather than preventing an otherwise valid chamber B-rep.
            pass
    cad = input.manufacturing.cad.lower()
    if cad in {CAD_STEP, CAD_BOTH}:
        files["step"] = export_step(
            contour["x"], contour["y"], out / "thrust_chamber_wall.step",
            input.manufacturing.angular_points,
            wall_thickness=input.manufacturing.wall_thickness,
            flange_od=input.interface.flange_od,
            flange_length=input.interface.flange_length,
            bolt_count=input.interface.bolt_count,
            bolt_circle_diameter=input.interface.bolt_circle_diameter,
            bolt_hole_diameter=input.interface.bolt_hole_diameter,
            seal_center_radius=seal_center,
            seal_groove_width=seal_width,
            # The injector face owns the O-ring gland; the chamber flange is
            # the continuous flat mating land.  Cutting two half-glands would
            # be an invalid default for a conventional static face O-ring.
            seal_groove_depth=None,
            metadata=metadata,
            throat_location=contour["throat_location"],
            require_brep=True,
        )
        cad_sidecar = files["step"].with_suffix(".cad.json")
        if cad_sidecar.exists():
            files["step_cad_metadata"] = cad_sidecar

    if cad in {CAD_STL, CAD_BOTH}:
        files["stl"] = export_stl(
            contour["x"], contour["y"], out / "thrust_chamber_wall.stl",
            input.manufacturing.angular_points,
            wall_thickness=input.manufacturing.wall_thickness,
            flange_od=input.interface.flange_od,
            flange_length=input.interface.flange_length,
        )

    report_payload = {
        "input": input,
        "metadata": metadata,
        "gate_report": gate_report.to_dict(),
        "report_sections": report_sections,
    }
    report_path = out / "design_report_v2.json"
    report_path.write_text(
        json.dumps(
            _json_ready(report_payload),
            indent=2,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    files["design_report"] = report_path
    return files


def _v2_metadata(
    input: DesignInput,
    contour: dict,
    gate_report: DesignGateReport,
    report_sections: dict[str, Any],
) -> dict[str, Any]:
    interface = input.interface
    manufacturing = input.manufacturing
    return _json_ready({
        "design_mode": input.mode,
        "design_status": contour.get("design_status"),
        "hardware_qualified": False,
        "qualification_note": contour.get("qualification_note"),
        "gate_passed": gate_report.passed,
        "software_gate_passed": gate_report.passed,
        "physical_release_evidence_complete": report_sections[
            "physical_release_readiness"
        ]["evidence_complete"],
        "physical_release_blocked": report_sections[
            "physical_release_readiness"
        ]["blocked"],
        "configuration_id": input.configuration_id,
        "authoritative_cad": (
            "STEP"
            if manufacturing.cad.lower() in {CAD_STEP, CAD_BOTH}
            else None
        ),
        "native_ipt": "deferred",
        "thermo_mode": input.thermo.mode,
        "thermo_source": report_sections["thermochemistry"]["source"],
        "nozzle_expansion_model": report_sections["thermochemistry"][
            "expansion_model"
        ],
        "frozen_flow_fingerprint_sha256": (
            report_sections["thermochemistry"].get("frozen_expansion") or {}
        ).get("input_fingerprint_sha256"),
        "cooling_method": input.cooling.method,
        "injector_type": input.injector.type,
        "injector_feasible": report_sections["injector"].get("feasible", True),
        "spray_cstar_coupling_enabled": report_sections[
            "spray_cstar_coupling"
        ].get("enabled", input.spray_cstar_coupling.enabled),
        "material": input.material.name,
        "wall_thickness": manufacturing.wall_thickness,
        "flange_od": interface.flange_od,
        "flange_length": interface.flange_length,
        "bolt_count": interface.bolt_count,
        "bolt_circle_diameter": interface.bolt_circle_diameter,
        "bolt_hole_diameter": interface.bolt_hole_diameter,
        "bolt_diameter": interface.bolt_diameter,
        "bolt_allowable_stress": interface.bolt_allowable_stress,
        "joint_separation_factor": interface.joint_separation_factor,
        "injector_face_od": interface.injector_face_od,
        "injector_face_thickness": interface.injector_face_thickness,
        "throat_insert": manufacturing.throat_insert,
        "throat_insert_material": manufacturing.throat_insert_material,
        "tolerance": manufacturing.tolerance,
        "weld_allowance": manufacturing.weld_allowance,
        "braze_allowance": manufacturing.braze_allowance,
        "authoritative_contour": "injector_face_to_chamber_to_throat_to_bell_exit",
        "cad_body_scope": "single_revolved_uniform_wall_body",
        "multi_body_cad_status": "deferred_until_liner_channel_jacket_interfaces_are_defined",
    })


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _json_ready(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        # JSON has no standard NaN/Infinity tokens.  Infinite internal
        # sentinels (for example a deliberately disabled screening limit)
        # remain meaningful in memory but are emitted as JSON null.
        return None
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    return value
