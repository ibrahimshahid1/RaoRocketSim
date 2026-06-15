"""
design.py - High-level design-gated nozzle workflow.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from raosim.cea import (
    THERMO_CEA_FROZEN,
    THERMO_CONSTANT_GAMMA,
    ThermochemistryResult,
    propellant_from_request,
    resolve_thermochemistry,
)
from raosim.engine import EnginePerformance, compute_engine_performance
from raosim.export import export_csv, export_step, export_stl, package_ipt_request
from raosim.gas_dynamics import (
    expansion_ratio_from_pressure,
    isentropic_pressure_ratio,
    mach_from_area_ratio,
    thrust_coefficient,
)
from raosim.nozzle_geometry import bell_nozzle_contour, lookup_angles
from raosim.physics import (
    bartz_heat_flux,
    boundary_layer_displacement,
    regenerative_cooling_screen,
    structural_screen,
)
from raosim.propellants import Propellant
from raosim.validation import DesignGateReport, evaluate_design_gates


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
    eta_Isp: float = 0.95


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
class CoolingSpec:
    method: str = "none"
    coolant: str | None = None
    channel_count: int | None = None
    channel_width: float | None = None
    channel_height: float | None = None
    coolant_mass_flow: float | None = None
    coolant_cp: float = 3500.0
    coolant_inlet_temperature: float = 293.0
    max_wall_temperature: float = 950.0
    # Optional coolant transport properties (override the built-in
    # COOLANT_PROPERTIES table keyed by ``coolant`` name).  Supply
    # CEA/measured values for accuracy; used by the Sieder-Tate solve.
    coolant_density: float | None = None        # kg/m^3
    coolant_viscosity: float | None = None      # Pa.s (disables Andrade T-model)
    coolant_conductivity: float | None = None   # W/(m.K)


@dataclass
class MaterialSpec:
    name: str = "Inconel 718"
    yield_strength: float = 900e6
    conductivity: float = 16.0
    max_temperature: float = 1250.0
    max_heat_flux: float = 25e6


@dataclass
class InterfaceSpec:
    flange_od: float | None = None
    flange_length: float | None = None
    bolt_count: int | None = None
    bolt_circle_diameter: float | None = None
    bolt_hole_diameter: float | None = None
    injector_face_od: float | None = None
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
    ambient: MissionAmbientSpec = field(default_factory=MissionAmbientSpec)
    cooling: CoolingSpec = field(default_factory=CoolingSpec)
    material: MaterialSpec = field(default_factory=MaterialSpec)
    interface: InterfaceSpec = field(default_factory=InterfaceSpec)
    manufacturing: ManufacturingSpec = field(default_factory=ManufacturingSpec)
    strict_gates: bool = False


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

    def to_dict(self) -> dict:
        return _json_ready({
            "mode": self.input.mode,
            "validated": self.validated,
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
                "Pe": self.performance.Pe,
                "Me": self.performance.Me,
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


def design_nozzle_v2(input: DesignInput) -> ValidatedDesignResult:
    """Generate a physics-screened nozzle design from the strict v2 schema."""
    _validate_design_input(input)
    require_cea = input.mode == DESIGN_MODE_VALIDATED

    thermo = resolve_thermochemistry(
        thermo_mode=input.thermo.mode,
        propellant_name=input.thermo.propellant_name,
        Pc=input.Pc,
        mixture_ratio=input.thermo.mixture_ratio,
        oxidizer=input.thermo.oxidizer,
        fuel=input.thermo.fuel,
        eta_Isp=input.thermo.eta_Isp,
        epsilon=input.epsilon,
        require_cea=require_cea,
    )
    prop = thermo.propellant
    warnings = list(thermo.warnings)

    epsilon = input.epsilon
    if epsilon is None:
        epsilon, _ = expansion_ratio_from_pressure(
            input.Pc, input.ambient.design_pressure, prop.gamma
        )
        input.epsilon = epsilon
        warnings.append(
            f"Sized epsilon from Pc/design Pa: epsilon = {epsilon:.3f}."
        )

    Rt = input.Rt
    if Rt is not None and input.target_thrust is not None:
        warnings.append("Both Rt and target_thrust supplied; explicit Rt is used.")
    if Rt is None:
        if input.target_thrust is None:
            raise ValueError("Either Rt or target_thrust must be provided.")
        Rt = throat_radius_for_target_thrust(
            input.target_thrust, input.Pc, input.ambient.design_pressure,
            float(epsilon), prop,
        )
        input.Rt = Rt
        warnings.append(f"Sized Rt from target thrust: Rt = {Rt * 1000:.3f} mm.")

    contour = _build_v2_contour(input, Rt, float(epsilon), prop)
    performance = compute_engine_performance(
        input.Pc, input.ambient.Pa, Rt, float(epsilon), prop
    )

    gate_report = evaluate_design_gates(
        contour,
        input.Pc,
        input.ambient.Pa,
        prop.gamma,
        wall_thickness=input.manufacturing.wall_thickness,
        flange_od=input.interface.flange_od,
        flange_length=input.interface.flange_length,
    )

    boundary_layer = boundary_layer_displacement(contour, input.Pc, prop)
    thermal = bartz_heat_flux(contour, input.Pc, prop)
    # Pass prop + Pc so the cooling screen runs the real coupled
    # Sieder-Tate / 1-D wall-conduction solve (gas side = full Bartz).
    cooling = regenerative_cooling_screen(
        thermal, contour, input.cooling, input.material,
        input.manufacturing.wall_thickness, prop, input.Pc,
    )
    structural = structural_screen(
        contour, input.Pc, input.ambient.Pa, prop, input.material,
        input.manufacturing.wall_thickness, thermal, cooling,
    )
    cad_readiness = _cad_readiness(input, contour, gate_report)
    benchmark_status = _benchmark_status(input.method)

    _add_v2_gate_checks(
        gate_report, input, thermo, boundary_layer, thermal, cooling,
        structural, cad_readiness, benchmark_status,
    )

    warnings.extend(contour.get("warnings", []))
    warnings.extend(gate_report.warnings)
    warnings.extend(cooling.get("warnings", []))
    report_sections = {
        "thermochemistry": {
            "mode": thermo.mode,
            "source": thermo.source,
            "cea_available": thermo.cea_available,
            "chamber_state": thermo.chamber_state,
            "exit_state": thermo.exit_state,
        },
        "boundary_layer": boundary_layer,
        "thermal": thermal,
        "cooling": cooling,
        "structural": structural,
        "cad_readiness": cad_readiness,
        "benchmark_status": benchmark_status,
    }

    if input.mode == DESIGN_MODE_VALIDATED and not gate_report.passed:
        raise RuntimeError(
            "Validated design gates failed: " + "; ".join(gate_report.warnings)
        )
    if input.strict_gates and not gate_report.passed:
        raise RuntimeError(
            "Design gates failed: " + "; ".join(gate_report.warnings)
        )

    files = _write_v2_artifacts(input, contour, gate_report, report_sections)
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
        theta_n = request.theta_n
        theta_e = request.theta_e
        if theta_n is None or theta_e is None:
            tn_l, te_l = lookup_angles(request.epsilon, request.length_pct)
            theta_n = theta_n if theta_n is not None else tn_l
            theta_e = theta_e if theta_e is not None else te_l
        contour = bell_nozzle_contour(
            Rt, request.epsilon, theta_n, theta_e,
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
) -> float:
    if target_thrust <= 0.0:
        raise ValueError("target_thrust must be positive")
    Me = mach_from_area_ratio(epsilon, prop.gamma, supersonic=True)
    pe_pc = isentropic_pressure_ratio(Me, prop.gamma)
    cf_ideal = thrust_coefficient(Me, prop.gamma, pe_pc, Pa / Pc, epsilon)
    cf_actual = cf_ideal * prop.eta_Isp
    if cf_actual <= 0.0:
        raise ValueError("target thrust cannot be met with non-positive Cf")
    At = target_thrust / (cf_actual * Pc)
    return math.sqrt(At / math.pi)


def _validate_design_input(input: DesignInput) -> None:
    if input.mode not in DESIGN_MODES:
        raise ValueError("mode must be 'preliminary' or 'validated'")
    if input.method not in {"bezier", "moc", "rao", "rao_variational_moc"}:
        raise ValueError("method must be one of: bezier, moc, rao, rao_variational_moc")
    if input.Pc <= 0.0:
        raise ValueError("Pc must be positive")
    if input.Rt is not None and input.Rt <= 0.0:
        raise ValueError("Rt must be positive when supplied")
    if input.target_thrust is not None and input.target_thrust <= 0.0:
        raise ValueError("target_thrust must be positive when supplied")
    if input.epsilon is not None and input.epsilon <= 1.0:
        raise ValueError("epsilon must be > 1 when supplied")
    if input.mode == DESIGN_MODE_VALIDATED:
        if input.method != "bezier":
            raise ValueError("validated mode only supports the benchmarked bezier path")
        if input.thermo.mode == THERMO_CONSTANT_GAMMA:
            raise RuntimeError("validated mode requires CEA thermochemistry")
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
    if (input.interface.flange_od is None) != (input.interface.flange_length is None):
        raise ValueError("flange_od and flange_length must be supplied together")
    if input.cooling.method not in {"none", "regenerative"}:
        raise ValueError("cooling.method must be 'none' or 'regenerative'")
    if input.cooling.method == "regenerative":
        if not input.cooling.channel_count or input.cooling.channel_count <= 0:
            raise ValueError("regenerative cooling requires channel_count > 0")
        if not input.cooling.channel_width or input.cooling.channel_width <= 0.0:
            raise ValueError("regenerative cooling requires channel_width > 0")
        if not input.cooling.channel_height or input.cooling.channel_height <= 0.0:
            raise ValueError("regenerative cooling requires channel_height > 0")
        if not input.cooling.coolant_mass_flow or input.cooling.coolant_mass_flow <= 0.0:
            raise ValueError("regenerative cooling requires coolant_mass_flow > 0")


def _build_v2_contour(
    input: DesignInput,
    Rt: float,
    epsilon: float,
    prop: Propellant,
) -> dict:
    if input.method == "bezier":
        theta_n = input.theta_n
        theta_e = input.theta_e
        if theta_n is None or theta_e is None:
            tn_l, te_l = lookup_angles(epsilon, input.length_pct)
            theta_n = theta_n if theta_n is not None else tn_l
            theta_e = theta_e if theta_e is not None else te_l
        return bell_nozzle_contour(
            Rt, epsilon, theta_n, theta_e, input.length_pct, gamma=prop.gamma
        )
    return bell_nozzle_contour(
        Rt, epsilon, method=input.method,
        length_pct=input.length_pct, gamma=prop.gamma,
    )


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
        float(structural["stress_margin"]) >= 1.5,
        value=float(structural["stress_margin"]), limit=">= 1.5",
        message="Thin-wall hoop stress screening margin is too low.",
    )
    report.add(
        "cad", "step_authoritative",
        bool(cad_readiness["step_authoritative"]),
        value=cad_readiness["requested_cad"], limit="STEP/STL with STEP authoritative",
        message="STEP remains the authoritative CAD artifact; native IPT is deferred.",
    )


def _cad_readiness(
    input: DesignInput,
    contour: dict,
    gate_report: DesignGateReport,
) -> dict:
    cad = input.manufacturing.cad.lower()
    max_od = 2.0 * float(max(contour["y"]))
    flange_ok = (
        input.interface.flange_od is None
        or (
            input.interface.flange_length is not None
            and input.interface.flange_od > max_od
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
        "placeholders": placeholders,
        "legacy_gate_passed_before_v2": gate_report.passed,
    }


def _benchmark_status(method: str) -> dict:
    if method == "bezier":
        return {
            "status": "benchmarked_preliminary_top_geometry",
            "validated_for_design": True,
            "notes": ["Bezier/TOP path is the trusted preliminary baseline."],
        }
    return {
        "status": "experimental_xfail_until_literature_benchmarks_pass",
        "validated_for_design": False,
        "notes": [
            "MOC/Rao paths remain diagnostic and blocked from manufacturing outputs."
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
        json.dumps(gate_report.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    files["design_report"] = report_path
    return files


def _write_v2_artifacts(
    input: DesignInput,
    contour: dict,
    gate_report: DesignGateReport,
    report_sections: dict[str, Any],
) -> dict[str, Path]:
    if input.manufacturing.output_dir is None:
        return {}

    out = Path(input.manufacturing.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    files["csv"] = export_csv(
        contour["x"], contour["y"], out / "nozzle_profile.csv",
        input.manufacturing.csv_points,
    )

    metadata = _v2_metadata(input, contour, gate_report, report_sections)
    cad = input.manufacturing.cad.lower()
    if cad in {CAD_STEP, CAD_BOTH}:
        files["step"] = export_step(
            contour["x"], contour["y"], out / "nozzle.step",
            input.manufacturing.angular_points,
            wall_thickness=input.manufacturing.wall_thickness,
            flange_od=input.interface.flange_od,
            flange_length=input.interface.flange_length,
            metadata=metadata,
        )

    if cad in {CAD_STL, CAD_BOTH}:
        files["stl"] = export_stl(
            contour["x"], contour["y"], out / "nozzle.stl",
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
        json.dumps(_json_ready(report_payload), indent=2) + "\n",
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
        "authoritative_cad": "STEP",
        "native_ipt": "deferred",
        "thermo_mode": input.thermo.mode,
        "thermo_source": report_sections["thermochemistry"]["source"],
        "cooling_method": input.cooling.method,
        "material": input.material.name,
        "wall_thickness": manufacturing.wall_thickness,
        "flange_od": interface.flange_od,
        "flange_length": interface.flange_length,
        "bolt_count": interface.bolt_count,
        "bolt_circle_diameter": interface.bolt_circle_diameter,
        "bolt_hole_diameter": interface.bolt_hole_diameter,
        "throat_insert": manufacturing.throat_insert,
        "throat_insert_material": manufacturing.throat_insert_material,
        "tolerance": manufacturing.tolerance,
        "weld_allowance": manufacturing.weld_allowance,
        "braze_allowance": manufacturing.braze_allowance,
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
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
