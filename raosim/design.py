"""
design.py - High-level design-gated nozzle workflow.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from raosim.cea import propellant_from_request
from raosim.engine import EnginePerformance, compute_engine_performance
from raosim.export import export_csv, export_step, export_stl, package_ipt_request
from raosim.gas_dynamics import isentropic_pressure_ratio, mach_from_area_ratio, thrust_coefficient
from raosim.nozzle_geometry import bell_nozzle_contour, lookup_angles
from raosim.propellants import Propellant
from raosim.validation import DesignGateReport, evaluate_design_gates


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


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result
