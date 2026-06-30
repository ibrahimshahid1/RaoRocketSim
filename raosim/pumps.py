"""Electric pump-feed sizing layered on the injector feed ledger.

The injector module already owns the feed-pressure balance.  This module
starts from that ``FeedSystemLedger`` and adds first-pass electric drive,
battery, centrifugal pump, inducer, diffuser/volute, and feasibility screens.
All geometry correlations here are preliminary sizing rules anchored to the
standard rocket turbopump design variables (head coefficient, suction specific
speed, specific speed/diameter), not hardware qualification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from raosim.injector import FeedSystemLedger, InjectorGate


G0 = 9.80665


SCREENING_DEFAULTS = {
    # These are technology/envelope assumptions used only when no component
    # data are supplied.  Pump efficiency and RPM are intentionally not fixed
    # here: the default path estimates them from the solved pump duty.
    "pump_margin": 0.10,
    "motor_efficiency": 0.90,
    "inverter_efficiency": 0.96,
    "motor_power_density": 2500.0,
    "inverter_power_density": 15000.0,
    "battery_energy_density": 250.0 * 3600.0,
    "battery_power_density": 3000.0,
    "battery_discharge_efficiency": 0.95,
    "battery_structural_margin": 1.20,
    "auto_current_target": 250.0,
}


LITERATURE_SOURCES = {
    "NASA SP-8109": (
        "Liquid Rocket Engine Centrifugal Flow Turbopumps; used for the "
        "flow/head/inlet-pressure dependency chain, head coefficient, "
        "specific-speed/specific-diameter screening, diffuser/volute choice, "
        "tip-speed/staging cautions, and small-pump scale cautions."
    ),
    "NASA SP-8052": (
        "Liquid Rocket Engine Turbopump Inducers; used for the NPSH, "
        "suction-specific-speed, inducer diameter, blade count, and solidity "
        "screening logic."
    ),
    "Lee et al. 2021": (
        "Performance Analysis and Mass Estimation of a Small-Sized Liquid "
        "Rocket Engine with Electric-Pump Cycle; used as a comparison case "
        "for electric-pump mass closure, not as universal pump RPM or pump "
        "efficiency defaults."
    ),
    "Spiller, Stabile, Lentini 2013": (
        "Design and Testing of a Demonstrator Electric Pump Feed System for "
        "Liquid Propellant Rocket Engines; used as a warning that very small "
        "off-the-shelf pumps can have poor efficiency and may require custom "
        "pump design."
    ),
}


def _default_pump_efficiencies() -> dict[str, float | None]:
    # Empty means "infer from pump duty"; explicit values remain supported.
    return {}


def _finite(x: float | None) -> bool:
    return x is not None and math.isfinite(float(x))


def _safe_sqrt(x: float) -> float:
    return math.sqrt(max(0.0, x))


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, x))


@dataclass
class ElectricDriveSpec:
    motor_efficiency: float = SCREENING_DEFAULTS["motor_efficiency"]
    inverter_efficiency: float = SCREENING_DEFAULTS["inverter_efficiency"]
    voltage: float | None = None
    rpm: float | None = None
    max_rpm: float = 120000.0
    max_motor_power: float | None = None
    max_current: float | None = None
    motor_power_density: float = SCREENING_DEFAULTS["motor_power_density"]
    inverter_power_density: float = SCREENING_DEFAULTS["inverter_power_density"]
    # Legacy combined motor+controller override, kept for API compatibility.
    power_density: float | None = None
    torque_limit: float | None = None
    heat_rejection_limit: float | None = None


@dataclass
class BatterySpec:
    energy_density: float = SCREENING_DEFAULTS["battery_energy_density"]
    power_density: float = SCREENING_DEFAULTS["battery_power_density"]
    discharge_efficiency: float = SCREENING_DEFAULTS["battery_discharge_efficiency"]
    structural_margin: float = SCREENING_DEFAULTS["battery_structural_margin"]
    voltage: float | None = None
    max_current: float | None = None
    max_mass_fraction: float | None = None
    vehicle_mass: float | None = None


@dataclass
class PumpSizingSpec:
    drive: ElectricDriveSpec = field(default_factory=ElectricDriveSpec)
    battery: BatterySpec = field(default_factory=BatterySpec)
    burn_time: float = 10.0
    pump_efficiency: dict[str, float | None] = field(default_factory=_default_pump_efficiencies)
    head_coefficient: float = 0.55
    flow_coefficient: float = 0.08
    inlet_flow_coefficient: float = 0.12
    blade_count: int = 6
    inducer_blade_count: int = 3
    inducer_solidity: float = 1.5
    inducer_hub_ratio: float = 0.35
    diffuser_vane_count: int = 8
    material_tip_speed_limit: float = 350.0
    max_head_per_stage: float = 2500.0
    target_specific_speed: float = 0.45
    min_impeller_diameter: float = 0.008
    max_impeller_diameter: float = 0.18
    min_outlet_width: float = 3.0e-4
    max_outlet_width_ratio: float = 0.12
    auto_current_target: float = SCREENING_DEFAULTS["auto_current_target"]
    min_capacity_margin: float = SCREENING_DEFAULTS["pump_margin"]


@dataclass
class ElectricDriveSizing:
    role: str
    rpm: float
    voltage: float
    voltage_source: str
    shaft_power: float
    electric_power: float
    torque: float
    motor_mass: float
    inverter_mass: float
    motor_controller_mass: float
    current: float
    controller_heat: float
    motor_heat: float
    total_heat: float

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "rpm": self.rpm,
            "voltage_v": self.voltage,
            "voltage_source": self.voltage_source,
            "shaft_power_w": self.shaft_power,
            "electric_power_w": self.electric_power,
            "torque_n_m": self.torque,
            "current_a": self.current,
            "motor_mass_kg": self.motor_mass,
            "inverter_mass_kg": self.inverter_mass,
            "motor_controller_mass_kg": self.motor_controller_mass,
            "controller_heat_w": self.controller_heat,
            "motor_heat_w": self.motor_heat,
            "total_heat_w": self.total_heat,
        }


@dataclass
class BatterySizing:
    electric_power: float
    burn_time: float
    voltage: float
    energy_required: float
    mass_energy_limited: float
    mass_power_limited: float
    mass: float
    current: float
    heat: float
    limiting: str
    vehicle_mass_fraction: float | None

    def to_dict(self) -> dict:
        return {
            "electric_power_w": self.electric_power,
            "burn_time_s": self.burn_time,
            "voltage_v": self.voltage,
            "energy_required_j": self.energy_required,
            "mass_energy_limited_kg": self.mass_energy_limited,
            "mass_power_limited_kg": self.mass_power_limited,
            "mass_kg": self.mass,
            "current_a": self.current,
            "heat_w": self.heat,
            "limiting": self.limiting,
            "vehicle_mass_fraction": self.vehicle_mass_fraction,
        }


@dataclass
class CentrifugalPumpGeometry:
    role: str
    rpm: float
    stages: int
    specific_speed: float
    specific_diameter: float
    head_coefficient: float
    flow_coefficient: float
    tip_speed: float
    impeller_diameter: float
    inlet_diameter: float
    outlet_width: float
    blade_count: int
    inlet_blade_angle_deg: float
    outlet_blade_angle_deg: float
    recommendation: str

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "rpm": self.rpm,
            "stages": self.stages,
            "specific_speed": self.specific_speed,
            "specific_diameter": self.specific_diameter,
            "head_coefficient": self.head_coefficient,
            "flow_coefficient": self.flow_coefficient,
            "tip_speed_m_s": self.tip_speed,
            "impeller_diameter_m": self.impeller_diameter,
            "inlet_diameter_m": self.inlet_diameter,
            "outlet_width_m": self.outlet_width,
            "blade_count": self.blade_count,
            "inlet_blade_angle_deg": self.inlet_blade_angle_deg,
            "outlet_blade_angle_deg": self.outlet_blade_angle_deg,
            "recommendation": self.recommendation,
        }


@dataclass
class InducerGeometry:
    role: str
    diameter: float
    hub_ratio: float
    blade_count: int
    solidity: float
    pitch: float
    wrap_angle_deg: float
    suction_specific_speed: float | None
    npsh_margin: float | None
    recommendation: str

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "diameter_m": self.diameter,
            "hub_ratio": self.hub_ratio,
            "blade_count": self.blade_count,
            "solidity": self.solidity,
            "pitch_m": self.pitch,
            "wrap_angle_deg": self.wrap_angle_deg,
            "suction_specific_speed": self.suction_specific_speed,
            "npsh_margin_pa": self.npsh_margin,
            "recommendation": self.recommendation,
        }


@dataclass
class DiffuserVoluteGeometry:
    role: str
    selection: str
    throat_area: float
    vane_count: int
    vane_width: float
    volute_exit_area: float
    diffusion_ratio: float
    recommendation: str

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "selection": self.selection,
            "throat_area_m2": self.throat_area,
            "vane_count": self.vane_count,
            "vane_width_m": self.vane_width,
            "volute_exit_area_m2": self.volute_exit_area,
            "diffusion_ratio": self.diffusion_ratio,
            "recommendation": self.recommendation,
        }


@dataclass
class PumpVelocityTriangle:
    """Single-point centrifugal-pump velocity triangle, per stage.

    Angles are measured from the tangential direction.  The inlet assumes no
    prewhirl; the outlet uses a Stodola-style slip screen so the exported
    geometry has an explicit bridge from blade angles to Euler head.
    """

    inlet_tip_speed: float
    outlet_tip_speed: float
    inlet_meridional_velocity: float
    outlet_meridional_velocity: float
    inlet_whirl_velocity: float
    outlet_whirl_velocity: float
    inlet_relative_velocity: float
    outlet_relative_velocity: float
    inlet_blade_angle_deg: float
    outlet_blade_angle_deg: float
    outlet_absolute_flow_angle_deg: float
    slip_factor: float
    euler_head: float
    required_stage_head: float
    euler_head_margin: float

    def to_dict(self) -> dict:
        return {
            "inlet_tip_speed_m_s": self.inlet_tip_speed,
            "outlet_tip_speed_m_s": self.outlet_tip_speed,
            "inlet_meridional_velocity_m_s": self.inlet_meridional_velocity,
            "outlet_meridional_velocity_m_s": self.outlet_meridional_velocity,
            "inlet_whirl_velocity_m_s": self.inlet_whirl_velocity,
            "outlet_whirl_velocity_m_s": self.outlet_whirl_velocity,
            "inlet_relative_velocity_m_s": self.inlet_relative_velocity,
            "outlet_relative_velocity_m_s": self.outlet_relative_velocity,
            "inlet_blade_angle_deg": self.inlet_blade_angle_deg,
            "outlet_blade_angle_deg": self.outlet_blade_angle_deg,
            "outlet_absolute_flow_angle_deg": self.outlet_absolute_flow_angle_deg,
            "slip_factor": self.slip_factor,
            "euler_head_m": self.euler_head,
            "required_stage_head_m": self.required_stage_head,
            "euler_head_margin_m": self.euler_head_margin,
        }


@dataclass
class PumpHydraulicLossBreakdown:
    reynolds_number: float | None
    incidence_loss_head: float
    blade_loading_loss_head: float
    passage_friction_loss_head: float
    disk_friction_loss_head: float
    leakage_loss_head: float
    recirculation_loss_head: float
    total_loss_head: float

    def to_dict(self) -> dict:
        return {
            "reynolds_number": self.reynolds_number,
            "incidence_loss_head_m": self.incidence_loss_head,
            "blade_loading_loss_head_m": self.blade_loading_loss_head,
            "passage_friction_loss_head_m": self.passage_friction_loss_head,
            "disk_friction_loss_head_m": self.disk_friction_loss_head,
            "leakage_loss_head_m": self.leakage_loss_head,
            "recirculation_loss_head_m": self.recirculation_loss_head,
            "total_loss_head_m": self.total_loss_head,
        }


@dataclass
class PumpHydraulicMeanline:
    role: str
    model: str
    source_ids: list[str]
    design_flow: float
    total_head: float
    stage_head: float
    stages: int
    hydraulic_efficiency: float
    efficiency_source: str
    velocity_triangle: PumpVelocityTriangle
    losses: PumpHydraulicLossBreakdown

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "model": self.model,
            "source_ids": self.source_ids,
            "design_flow_m3_s": self.design_flow,
            "total_head_m": self.total_head,
            "stage_head_m": self.stage_head,
            "stages": self.stages,
            "hydraulic_efficiency": self.hydraulic_efficiency,
            "efficiency_source": self.efficiency_source,
            "velocity_triangle": self.velocity_triangle.to_dict(),
            "losses": self.losses.to_dict(),
        }


@dataclass
class PumpCurvePoint:
    flow_ratio: float
    volumetric_flow: float
    head: float
    pressure_rise: float
    hydraulic_efficiency: float
    hydraulic_power: float
    shaft_power: float

    def to_dict(self) -> dict:
        return {
            "flow_ratio": self.flow_ratio,
            "volumetric_flow_m3_s": self.volumetric_flow,
            "head_m": self.head,
            "pressure_rise_pa": self.pressure_rise,
            "hydraulic_efficiency": self.hydraulic_efficiency,
            "hydraulic_power_w": self.hydraulic_power,
            "shaft_power_w": self.shaft_power,
        }


@dataclass
class PumpPerformanceCurve:
    role: str
    rpm: float
    source_ids: list[str]
    points: list[PumpCurvePoint]

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "rpm": self.rpm,
            "source_ids": self.source_ids,
            "points": [p.to_dict() for p in self.points],
        }


@dataclass
class PumpLineSizing:
    role: str
    pressure_rise: float | None
    head: float | None
    volumetric_flow: float
    hydraulic_power: float | None
    shaft_power: float | None
    efficiency: float
    efficiency_source: str
    rpm_source: str | None
    drive: ElectricDriveSizing | None
    impeller: CentrifugalPumpGeometry | None
    inducer: InducerGeometry | None
    diffuser_volute: DiffuserVoluteGeometry | None
    hydraulic_meanline: PumpHydraulicMeanline | None = None
    performance_curve: PumpPerformanceCurve | None = None

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "required_pressure_rise_pa": self.pressure_rise,
            "required_pump_head_m": self.head,
            "volumetric_flow_m3_s": self.volumetric_flow,
            "hydraulic_power_w": self.hydraulic_power,
            "shaft_power_w": self.shaft_power,
            "pump_efficiency": self.efficiency,
            "pump_efficiency_source": self.efficiency_source,
            "rpm_source": self.rpm_source,
            "drive": self.drive.to_dict() if self.drive else None,
            "impeller": self.impeller.to_dict() if self.impeller else None,
            "inducer": self.inducer.to_dict() if self.inducer else None,
            "diffuser_volute": (
                self.diffuser_volute.to_dict() if self.diffuser_volute else None
            ),
            "hydraulic_meanline": (
                self.hydraulic_meanline.to_dict()
                if self.hydraulic_meanline else None
            ),
            "performance_curve": (
                self.performance_curve.to_dict()
                if self.performance_curve else None
            ),
        }


@dataclass
class PumpFeasibility:
    feasible: bool
    gates: list[InjectorGate]
    suggestions: list[str]

    def to_dict(self) -> dict:
        return {
            "feasible": self.feasible,
            "gates": [
                {"name": g.name, "status": g.status, "detail": g.detail}
                for g in self.gates
            ],
            "suggestions": self.suggestions,
        }


@dataclass
class ElectricPumpSizingResult:
    feasible: bool
    lines: dict[str, PumpLineSizing]
    battery: BatterySizing
    feasibility: PumpFeasibility
    assumptions: dict
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "feasible": self.feasible,
            "lines": {k: v.to_dict() for k, v in self.lines.items()},
            "battery": self.battery.to_dict(),
            "feasibility": self.feasibility.to_dict(),
            "assumptions": self.assumptions,
            "notes": self.notes,
        }


def _estimate_pump_efficiency(Q: float, head: float | None) -> tuple[float, str]:
    """Return a screening pump efficiency from the solved hydraulic duty.

    Small rocket pumps are often low-specific-speed, low-flow machines; using a
    large-pump efficiency by default is as misleading as hard-coding one paper's
    RPM.  This curve is intentionally conservative and only supplies a first
    pass until a real pump curve is attached.
    """
    if not _finite(Q) or Q <= 0.0 or head is None or not _finite(head):
        return 0.35, "auto_screen:no_duty"
    Q = float(Q)
    head = max(float(head), 0.0)
    if Q < 2.0e-5:
        eta = 0.30
    elif Q < 1.0e-4:
        eta = 0.40
    elif Q < 5.0e-4:
        eta = 0.52
    elif Q < 2.0e-3:
        eta = 0.62
    else:
        eta = 0.70
    if head > 2500.0:
        eta -= 0.03
    if head > 6000.0:
        eta -= 0.04
    return min(0.78, max(0.25, eta)), "auto_screen:flow_head"


def _line_efficiency(
    spec: PumpSizingSpec,
    role: str,
    Q: float,
    head: float | None,
) -> tuple[float, str]:
    eta = spec.pump_efficiency.get(role)
    if eta is None:
        eta = spec.pump_efficiency.get("default")
    if eta is not None:
        return min(0.90, max(0.05, float(eta))), "user"
    return _estimate_pump_efficiency(Q, head)


def _screen_status(value: float, warn: float, fail: float, *, high_bad=True) -> str:
    if high_bad:
        if value >= fail:
            return "fail"
        if value >= warn:
            return "warn"
    else:
        if value <= fail:
            return "fail"
        if value <= warn:
            return "warn"
    return "pass"


def _select_voltage(electric_power: float, spec: PumpSizingSpec) -> tuple[float, str]:
    drive = spec.drive
    if drive.voltage is not None:
        return float(drive.voltage), "user"
    current_target = (
        drive.max_current
        or spec.battery.max_current
        or max(spec.auto_current_target, 1.0)
    )
    required = electric_power / max(current_target, 1e-9)
    for candidate in (24.0, 48.0, 96.0, 120.0, 270.0, 400.0, 540.0, 800.0):
        if candidate >= required:
            return candidate, "auto_standard_bus"
    return required, "auto_minimum_for_current_limit"


def _drive_sizing(
    role: str,
    shaft_power: float,
    rpm: float,
    spec: PumpSizingSpec,
) -> ElectricDriveSizing:
    drive = spec.drive
    eta_m = max(1e-6, min(1.0, drive.motor_efficiency))
    eta_i = max(1e-6, min(1.0, drive.inverter_efficiency))
    motor_input_power = shaft_power / eta_m
    electric_power = motor_input_power / eta_i
    omega = 2.0 * math.pi * max(rpm, 1e-9) / 60.0
    voltage, voltage_source = _select_voltage(electric_power, spec)
    current = electric_power / max(voltage, 1e-9)
    if drive.power_density is not None:
        motor_controller_mass = electric_power / max(drive.power_density, 1e-9)
        motor_mass = motor_controller_mass
        inverter_mass = 0.0
    else:
        motor_mass = shaft_power / max(drive.motor_power_density, 1e-9)
        inverter_mass = electric_power / max(drive.inverter_power_density, 1e-9)
        motor_controller_mass = motor_mass + inverter_mass
    return ElectricDriveSizing(
        role=role,
        rpm=rpm,
        voltage=voltage,
        voltage_source=voltage_source,
        shaft_power=shaft_power,
        electric_power=electric_power,
        torque=shaft_power / omega,
        motor_mass=motor_mass,
        inverter_mass=inverter_mass,
        motor_controller_mass=motor_controller_mass,
        current=current,
        controller_heat=electric_power - motor_input_power,
        motor_heat=motor_input_power - shaft_power,
        total_heat=electric_power - shaft_power,
    )


def _battery_sizing(
    total_electric_power: float,
    spec: PumpSizingSpec,
    selected_bus_voltage: float,
) -> BatterySizing:
    batt = spec.battery
    burn = max(0.0, spec.burn_time)
    eta = max(1e-6, min(1.0, batt.discharge_efficiency))
    energy_required = total_electric_power * burn / eta
    mass_energy = energy_required / max(batt.energy_density, 1e-9)
    mass_power = total_electric_power / max(batt.power_density, 1e-9)
    limiting = "energy" if mass_energy >= mass_power else "power"
    mass = max(mass_energy, mass_power) * max(1.0, batt.structural_margin)
    voltage = batt.voltage or spec.drive.voltage or selected_bus_voltage
    current = total_electric_power / max(voltage, 1e-9)
    heat = total_electric_power * (1.0 / eta - 1.0)
    mass_fraction = None
    if batt.vehicle_mass is not None and batt.vehicle_mass > 0.0:
        mass_fraction = mass / batt.vehicle_mass
    return BatterySizing(
        electric_power=total_electric_power,
        burn_time=burn,
        voltage=voltage,
        energy_required=energy_required,
        mass_energy_limited=mass_energy,
        mass_power_limited=mass_power,
        mass=mass,
        current=current,
        heat=heat,
        limiting=limiting,
        vehicle_mass_fraction=mass_fraction,
    )


def _stage_count(head: float, spec: PumpSizingSpec) -> int:
    return max(1, int(math.ceil(max(head, 0.0) / max(spec.max_head_per_stage, 1e-9))))


def _select_rpm(Q: float, head: float, spec: PumpSizingSpec) -> tuple[float, str]:
    if spec.drive.rpm is not None:
        return float(spec.drive.rpm), "user"
    if Q <= 0.0 or head <= 0.0:
        return min(30000.0, spec.drive.max_rpm), "auto_degenerate"

    stages = _stage_count(head, spec)
    stage_head = max(head, 0.0) / stages
    psi = max(spec.head_coefficient, 1e-6)
    phi = max(spec.flow_coefficient, 1e-6)
    tip_speed = _safe_sqrt(G0 * stage_head / psi)

    # Primary solve: choose RPM from nondimensional specific speed.
    target_ns = min(1.20, max(0.10, spec.target_specific_speed))
    omega_ns = target_ns * max((G0 * stage_head) ** 0.75, 1e-12) / max(math.sqrt(Q), 1e-12)
    rpm = omega_ns * 60.0 / (2.0 * math.pi)

    # Geometry bounds convert to RPM bounds because D2 = 2U/omega.
    rpm_min = 60.0 * tip_speed / (math.pi * max(spec.max_impeller_diameter, 1e-9))
    rpm_max = 60.0 * tip_speed / (math.pi * max(spec.min_impeller_diameter, 1e-9))

    # Avoid an outlet passage thinner than the manufacturability floor.
    d2_max_from_width = Q / max(math.pi * phi * tip_speed * spec.min_outlet_width, 1e-12)
    if d2_max_from_width > 0.0:
        rpm_min = max(rpm_min, 60.0 * tip_speed / (math.pi * d2_max_from_width))

    # Avoid a comically wide/low-speed impeller passage.
    d2_min_from_width_ratio = _safe_sqrt(
        Q / max(math.pi * phi * tip_speed * spec.max_outlet_width_ratio, 1e-12)
    )
    if d2_min_from_width_ratio > 0.0:
        rpm_max = min(rpm_max, 60.0 * tip_speed / (math.pi * d2_min_from_width_ratio))

    rpm_max = min(rpm_max, spec.drive.max_rpm)
    if rpm_min > rpm_max:
        rpm = min(max(rpm, min(rpm_min, spec.drive.max_rpm)), spec.drive.max_rpm)
        return rpm, "auto_specific_speed_limited_by_conflicting_geometry"
    return min(max(rpm, rpm_min), rpm_max), "auto_specific_speed_geometry"


def _impeller_geometry(
    role: str,
    Q: float,
    head: float,
    rpm: float,
    spec: PumpSizingSpec,
) -> CentrifugalPumpGeometry:
    rpm = max(rpm, 1e-9)
    omega = 2.0 * math.pi * rpm / 60.0
    stages = _stage_count(head, spec)
    stage_head = max(head, 0.0) / stages
    psi = max(spec.head_coefficient, 1e-6)
    phi = max(spec.flow_coefficient, 1e-6)
    # SP-8109 meanline screen: psi = gH/U2^2, phi = Cm2/U2,
    # Ns = omega*sqrt(Q)/(gH)^0.75, Ds = D2*(gH)^0.25/sqrt(Q).
    tip_speed = _safe_sqrt(G0 * stage_head / psi)
    d2 = 2.0 * tip_speed / omega
    b2 = Q / max(math.pi * d2 * phi * tip_speed, 1e-12)
    d1 = _safe_sqrt(4.0 * Q / max(math.pi * spec.inlet_flow_coefficient * tip_speed, 1e-12))
    ns = omega * math.sqrt(max(Q, 0.0)) / max((G0 * stage_head) ** 0.75, 1e-12)
    ds = d2 * (G0 * stage_head) ** 0.25 / max(math.sqrt(max(Q, 1e-18)), 1e-12)
    outlet_angle = 25.0 if stage_head > 0.0 else 0.0
    inlet_angle = math.degrees(math.atan2(phi, 1.0))
    recommendation = "shrouded radial impeller"
    if tip_speed > 0.80 * spec.material_tip_speed_limit:
        recommendation = "split head across stages or use higher-strength impeller"
    elif Q < 2.0e-4:
        recommendation = "miniature closed/semi-open impeller; expect efficiency loss"
    elif ns > 0.9:
        recommendation = "mixed-flow tendency; revisit centrifugal-only assumption"
    return CentrifugalPumpGeometry(
        role=role,
        rpm=rpm,
        stages=stages,
        specific_speed=ns,
        specific_diameter=ds,
        head_coefficient=psi,
        flow_coefficient=phi,
        tip_speed=tip_speed,
        impeller_diameter=d2,
        inlet_diameter=d1,
        outlet_width=b2,
        blade_count=spec.blade_count,
        inlet_blade_angle_deg=inlet_angle,
        outlet_blade_angle_deg=outlet_angle,
        recommendation=recommendation,
    )


def _slip_factor(blade_count: int, outlet_blade_angle_deg: float) -> float:
    """Stodola-style centrifugal impeller slip screen.

    The intent is not to replace blade-to-blade analysis; it makes the
    first-order Euler head depend on blade count and outlet angle instead of
    leaving the impeller as diameter-only geometry.
    """
    z = max(int(blade_count), 1)
    beta = math.radians(_clamp(outlet_blade_angle_deg, 5.0, 80.0))
    return _clamp(1.0 - math.pi * math.sin(beta) / z, 0.55, 0.92)


def _velocity_triangle(
    Q: float,
    head: float,
    impeller: CentrifugalPumpGeometry,
    spec: PumpSizingSpec,
) -> PumpVelocityTriangle:
    stages = max(impeller.stages, 1)
    stage_head = max(head, 0.0) / stages
    omega = 2.0 * math.pi * max(impeller.rpm, 1e-9) / 60.0
    u2 = max(impeller.tip_speed, 1e-9)
    u1 = omega * max(impeller.inlet_diameter, 0.0) / 2.0
    area1 = math.pi * max(impeller.inlet_diameter, 1e-12) ** 2 / 4.0
    cm1 = max(Q, 0.0) / max(area1, 1e-12)
    cm2 = max(spec.flow_coefficient * u2, 0.0)
    beta2 = math.radians(_clamp(impeller.outlet_blade_angle_deg, 5.0, 80.0))
    slip = _slip_factor(impeller.blade_count, impeller.outlet_blade_angle_deg)
    cu2 = max(0.0, slip * u2 - cm2 / max(math.tan(beta2), 1e-12))
    euler_head = u2 * cu2 / G0
    w1 = math.hypot(cm1, u1)
    w2 = math.hypot(cm2, max(u2 - cu2, 0.0))
    beta1 = math.degrees(math.atan2(cm1, max(u1, 1e-12)))
    alpha2 = math.degrees(math.atan2(cm2, max(cu2, 1e-12)))
    return PumpVelocityTriangle(
        inlet_tip_speed=u1,
        outlet_tip_speed=u2,
        inlet_meridional_velocity=cm1,
        outlet_meridional_velocity=cm2,
        inlet_whirl_velocity=0.0,
        outlet_whirl_velocity=cu2,
        inlet_relative_velocity=w1,
        outlet_relative_velocity=w2,
        inlet_blade_angle_deg=beta1,
        outlet_blade_angle_deg=impeller.outlet_blade_angle_deg,
        outlet_absolute_flow_angle_deg=alpha2,
        slip_factor=slip,
        euler_head=euler_head,
        required_stage_head=stage_head,
        euler_head_margin=euler_head - stage_head,
    )


def _pump_reynolds_number(ln, impeller: CentrifugalPumpGeometry) -> float | None:
    mu = getattr(ln, "viscosity", None)
    rho = getattr(ln, "density", None)
    if not (_finite(mu) and _finite(rho)) or mu <= 0.0 or rho <= 0.0:
        return None
    return rho * impeller.tip_speed * impeller.impeller_diameter / mu


def _hydraulic_meanline(
    role: str,
    Q: float,
    head: float,
    ln,
    impeller: CentrifugalPumpGeometry,
    spec: PumpSizingSpec,
) -> PumpHydraulicMeanline:
    triangle = _velocity_triangle(Q, head, impeller, spec)
    stage_head = triangle.required_stage_head
    re = _pump_reynolds_number(ln, impeller)

    if re is None:
        friction_fraction = 0.045
    elif re < 2.0e4:
        friction_fraction = 0.120
    elif re < 1.0e5:
        friction_fraction = 0.070
    elif re < 5.0e5:
        friction_fraction = 0.040
    else:
        friction_fraction = 0.028

    loading = stage_head / max(triangle.euler_head, 1e-9)
    loading_loss = stage_head * max(0.0, loading - 0.78) * 0.45
    if triangle.euler_head < stage_head:
        loading_loss += stage_head - triangle.euler_head

    incidence_loss = stage_head * (
        0.006 if 5.0 <= triangle.inlet_blade_angle_deg <= 35.0 else 0.025
    )
    passage_loss = stage_head * friction_fraction
    disk_loss = min(0.10 * stage_head, 0.004 * impeller.tip_speed**2 / G0)
    leakage_loss = 0.025 * stage_head

    ns = impeller.specific_speed
    recirc_fraction = 0.018
    if Q < 1.0e-4:
        recirc_fraction += 0.090
    elif Q < 5.0e-4:
        recirc_fraction += 0.055
    elif Q < 2.0e-3:
        recirc_fraction += 0.030
    if ns < 0.15 or ns > 1.20:
        recirc_fraction += 0.045
    elif ns < 0.25 or ns > 0.95:
        recirc_fraction += 0.020
    recirc_loss = stage_head * recirc_fraction

    total_loss = (
        incidence_loss + loading_loss + passage_loss + disk_loss
        + leakage_loss + recirc_loss
    )
    eta = _clamp(stage_head / max(stage_head + total_loss, 1e-9), 0.20, 0.84)
    losses = PumpHydraulicLossBreakdown(
        reynolds_number=re,
        incidence_loss_head=incidence_loss,
        blade_loading_loss_head=loading_loss,
        passage_friction_loss_head=passage_loss,
        disk_friction_loss_head=disk_loss,
        leakage_loss_head=leakage_loss,
        recirculation_loss_head=recirc_loss,
        total_loss_head=total_loss,
    )
    return PumpHydraulicMeanline(
        role=role,
        model="centrifugal_meanline_v1",
        source_ids=["NASA SP-8109", "NASA SP-8052"],
        design_flow=Q,
        total_head=head,
        stage_head=stage_head,
        stages=impeller.stages,
        hydraulic_efficiency=eta,
        efficiency_source="meanline_loss_model:SP-8109",
        velocity_triangle=triangle,
        losses=losses,
    )


def _pump_performance_curve(
    role: str,
    Q_design: float,
    head_design: float,
    density: float,
    rpm: float,
    eta_design: float,
) -> PumpPerformanceCurve:
    points: list[PumpCurvePoint] = []
    for flow_ratio in (0.50, 0.70, 0.85, 1.00, 1.15, 1.30):
        q = flow_ratio
        Q = max(0.0, Q_design * q)
        # Fixed-speed centrifugal screen: shutoff head above design, falling
        # head at high flow.  This is a curve generator for trades, not a
        # substitute for a tested pump map.
        head_ratio = _clamp(1.0 + 0.30 * (1.0 - q) - 0.55 * (q * q - 1.0),
                            0.05, 1.65)
        head = max(0.0, head_design * head_ratio)
        eta_shape = 1.0 - 1.65 * (q - 1.0) ** 2
        if q < 0.65:
            eta_shape -= 0.12
        eta = _clamp(eta_design * eta_shape, 0.10, eta_design)
        pressure = density * G0 * head
        hydraulic_power = pressure * Q
        shaft_power = hydraulic_power / max(eta, 1e-9)
        points.append(PumpCurvePoint(
            flow_ratio=q,
            volumetric_flow=Q,
            head=head,
            pressure_rise=pressure,
            hydraulic_efficiency=eta,
            hydraulic_power=hydraulic_power,
            shaft_power=shaft_power,
        ))
    return PumpPerformanceCurve(
        role=role,
        rpm=rpm,
        source_ids=["NASA SP-8109"],
        points=points,
    )


def _inducer_geometry(role: str, Q: float, ln, impeller: CentrifugalPumpGeometry, spec: PumpSizingSpec) -> InducerGeometry:
    rpm = max(impeller.rpm, 1e-9)
    omega = 2.0 * math.pi * rpm / 60.0
    npsh_head = None
    suction_ns = None
    if _finite(ln.npsh_available):
        npsh_head = max(float(ln.npsh_available), 0.0) / max(ln.density * G0, 1e-12)
    if npsh_head is not None and npsh_head > 0.0:
        # SP-8052 suction screen: Nss = omega*sqrt(Q)/(g*NPSH)^0.75.
        suction_ns = omega * math.sqrt(max(Q, 0.0)) / max((G0 * npsh_head) ** 0.75, 1e-12)
    eye_d = max(impeller.inlet_diameter, 0.45 * impeller.impeller_diameter)
    pitch = math.pi * eye_d / max(spec.inducer_blade_count, 1)
    wrap = 360.0 * spec.inducer_solidity / max(spec.inducer_blade_count, 1)
    recommendation = "axial inducer ahead of impeller eye"
    if ln.npsh_margin is not None and ln.npsh_margin < 0.0:
        recommendation = "negative NPSH margin; raise tank pressure/subcooling or lower inlet losses"
    elif suction_ns is not None and suction_ns > 4.0:
        recommendation = "high suction specific speed; reduce rpm or add inlet boost/inducer margin"
    return InducerGeometry(
        role=role,
        diameter=eye_d,
        hub_ratio=spec.inducer_hub_ratio,
        blade_count=spec.inducer_blade_count,
        solidity=spec.inducer_solidity,
        pitch=pitch,
        wrap_angle_deg=wrap,
        suction_specific_speed=suction_ns,
        npsh_margin=ln.npsh_margin,
        recommendation=recommendation,
    )


def _diffuser_volute_geometry(role: str, Q: float, impeller: CentrifugalPumpGeometry, spec: PumpSizingSpec) -> DiffuserVoluteGeometry:
    cm2 = max(spec.flow_coefficient * impeller.tip_speed, 1e-9)
    area_impeller_exit = 2.0 * math.pi * (0.5 * impeller.impeller_diameter) * max(impeller.outlet_width, 1e-9)
    throat_area = max(1.15 * Q / cm2, 0.35 * area_impeller_exit)
    vane_width = max(impeller.outlet_width * 1.15, 1e-6)
    volute_exit_area = max(1.8 * throat_area, Q / max(0.35 * impeller.tip_speed, 1e-9))
    if impeller.specific_speed < 0.20:
        selection = "volute"
        recommendation = "low specific speed; compact volute or vaneless diffuser"
        vane_count = 0
    else:
        selection = "vaned_diffuser"
        recommendation = "vaned diffuser with collecting volute"
        vane_count = spec.diffuser_vane_count
    return DiffuserVoluteGeometry(
        role=role,
        selection=selection,
        throat_area=throat_area,
        vane_count=vane_count,
        vane_width=vane_width,
        volute_exit_area=volute_exit_area,
        diffusion_ratio=volute_exit_area / max(throat_area, 1e-12),
        recommendation=recommendation,
    )


def _requirement_message() -> str:
    return (
        "Electric pump feed is infeasible for this Pc under current assumptions; "
        "reduce Pc, reduce injector/feed losses, increase tank pressure/NPSH, "
        "split motors, or consider a turbopump."
    )


def _feasibility(
    ledger: FeedSystemLedger,
    lines: dict[str, PumpLineSizing],
    battery: BatterySizing,
    spec: PumpSizingSpec,
) -> PumpFeasibility:
    gates: list[InjectorGate] = []
    suggestions: list[str] = []
    drive = spec.drive
    batt = spec.battery

    for role, sizing in lines.items():
        ln = ledger.lines[role]
        if sizing.pressure_rise is None:
            gates.append(InjectorGate(
                f"electric_pump_pressure_rise_{role}", "info",
                f"{role} needs {ln.required_outlet_pressure/1e5:.1f} bar pump "
                "outlet; add tank pressure to compute pump rise/head/power",
            ))
            continue
        assert sizing.shaft_power is not None
        assert sizing.drive is not None
        assert sizing.impeller is not None
        assert sizing.inducer is not None
        assert sizing.diffuser_volute is not None

        gates.append(InjectorGate(
            f"electric_pump_power_{role}",
            _screen_status(sizing.shaft_power, 0.60 * (drive.max_motor_power or float("inf")),
                           drive.max_motor_power or float("inf")),
            f"{role} shaft power {sizing.shaft_power/1000:.2f} kW "
            f"(electric {sizing.drive.electric_power/1000:.2f} kW)",
        ))
        gates.append(InjectorGate(
            f"electric_pump_power_scale_{role}",
            _screen_status(sizing.shaft_power, 100.0e3, 500.0e3),
            f"{role} shaft power scale {sizing.shaft_power/1000:.2f} kW "
            "(default screen warns above 100 kW per stream)",
        ))
        if drive.max_motor_power is not None and sizing.shaft_power > drive.max_motor_power:
            suggestions.append(f"{role}: split the pump across motors or lower pressure rise")

        gates.append(InjectorGate(
            f"electric_pump_rpm_{role}",
            "pass" if sizing.drive.rpm <= drive.max_rpm else "fail",
            f"{role} pump speed {sizing.drive.rpm:.0f} rpm vs limit "
            f"{drive.max_rpm:.0f} rpm ({sizing.rpm_source})",
        ))
        if drive.torque_limit is not None:
            gates.append(InjectorGate(
                f"electric_pump_torque_{role}",
                "pass" if sizing.drive.torque <= drive.torque_limit else "fail",
                f"{role} torque {sizing.drive.torque:.3g} N m vs limit "
                f"{drive.torque_limit:.3g} N m",
            ))
        if drive.max_current is not None:
            gates.append(InjectorGate(
                f"electric_drive_current_{role}",
                "pass" if sizing.drive.current <= drive.max_current else "fail",
                f"{role} phase/DC current screen {sizing.drive.current:.0f} A vs "
                f"limit {drive.max_current:.0f} A at {sizing.drive.voltage:.0f} V",
            ))
        if drive.heat_rejection_limit is not None:
            gates.append(InjectorGate(
                f"electric_drive_heat_{role}",
                "pass" if sizing.drive.total_heat <= drive.heat_rejection_limit else "fail",
                f"{role} drive heat {sizing.drive.total_heat:.0f} W vs limit "
                f"{drive.heat_rejection_limit:.0f} W",
            ))

        tip_status = _screen_status(
            sizing.impeller.tip_speed,
            0.85 * spec.material_tip_speed_limit,
            spec.material_tip_speed_limit,
        )
        gates.append(InjectorGate(
            f"impeller_tip_speed_{role}", tip_status,
            f"{role} tip speed {sizing.impeller.tip_speed:.0f} m/s vs "
            f"screen {spec.material_tip_speed_limit:.0f} m/s",
        ))
        if tip_status != "pass":
            suggestions.append(f"{role}: increase stages or diameter/material margin")

        ns = sizing.impeller.specific_speed
        ns_status = "pass" if 0.10 <= ns <= 1.20 else ("warn" if 0.05 <= ns <= 1.60 else "fail")
        gates.append(InjectorGate(
            f"centrifugal_specific_speed_{role}", ns_status,
            f"{role} nondimensional specific speed Ns={ns:.3f}; centrifugal "
            "screening band is roughly 0.10-1.20",
        ))
        ds = sizing.impeller.specific_diameter
        ds_status = "pass" if 1.5 <= ds <= 12.0 else ("warn" if 0.8 <= ds <= 18.0 else "fail")
        gates.append(InjectorGate(
            f"centrifugal_specific_diameter_{role}", ds_status,
            f"{role} nondimensional specific diameter Ds={ds:.2f}; screening "
            "band is roughly 1.5-12 for radial centrifugal duty",
        ))
        stages = sizing.impeller.stages
        stage_status = "pass" if stages == 1 else ("warn" if stages <= 3 else "fail")
        gates.append(InjectorGate(
            f"pump_stage_count_{role}", stage_status,
            f"{role} head requires {stages} centrifugal stage(s) with "
            f"{spec.max_head_per_stage:.0f} m/stage screen",
        ))

        if sizing.impeller.outlet_width < 3.0e-4:
            gates.append(InjectorGate(
                f"impeller_width_{role}", "warn",
                f"{role} outlet width {sizing.impeller.outlet_width*1e3:.2f} mm; "
                "mini pump scale effects will dominate efficiency",
            ))

        if ln.capacity_margin is not None:
            cap_frac = ln.capacity_margin / max(ln.volumetric_flow * ln.density, 1e-12)
            gates.append(InjectorGate(
                f"pump_capacity_margin_{role}",
                "pass" if cap_frac >= spec.min_capacity_margin else "warn",
                f"{role} mass-flow capacity margin {cap_frac*100:.1f}% "
                f"(target {spec.min_capacity_margin*100:.0f}%)",
            ))

        if ln.npsh_margin is None:
            gates.append(InjectorGate(
                f"inducer_npsh_{role}", "info",
                f"{role} NPSH margin unknown; supply tank pressure and NPSH requirement",
            ))
        else:
            gates.append(InjectorGate(
                f"inducer_npsh_{role}",
                "pass" if ln.npsh_margin >= 0.0 else "fail",
                f"{role} NPSH margin {ln.npsh_margin/1e5:+.2f} bar",
            ))
        ss = sizing.inducer.suction_specific_speed
        if ss is not None:
            gates.append(InjectorGate(
                f"inducer_suction_specific_speed_{role}",
                "pass" if ss <= 4.0 else ("warn" if ss <= 6.0 else "fail"),
                f"{role} suction specific speed {ss:.2f}; high values point to "
                "cavitation risk",
            ))

        if (sizing.diffuser_volute.selection == "volute"
                and sizing.impeller.specific_speed > 0.70):
            gates.append(InjectorGate(
                f"diffuser_selection_{role}", "warn",
                f"{role} volute-only selection conflicts with Ns={ns:.2f}; "
                "consider vaned diffusion",
            ))

    if batt.max_current is not None:
        gates.append(InjectorGate(
            "battery_current", "pass" if battery.current <= batt.max_current else "fail",
            f"pack current {battery.current:.0f} A vs limit {batt.max_current:.0f} A",
        ))
    if batt.max_mass_fraction is not None and battery.vehicle_mass_fraction is not None:
        gates.append(InjectorGate(
            "battery_mass_fraction",
            "pass" if battery.vehicle_mass_fraction <= batt.max_mass_fraction else "fail",
            f"battery mass fraction {battery.vehicle_mass_fraction*100:.1f}% vs "
            f"limit {batt.max_mass_fraction*100:.1f}%",
        ))
    elif battery.vehicle_mass_fraction is not None:
        gates.append(InjectorGate(
            "battery_mass_fraction", "warn" if battery.vehicle_mass_fraction > 0.20 else "pass",
            f"battery mass fraction {battery.vehicle_mass_fraction*100:.1f}% "
            "(screening default warns above 20%)",
        ))
    gates.append(InjectorGate(
        "battery_power_energy_balance",
        "warn" if battery.mass_power_limited > 2.0 * battery.mass_energy_limited else "pass",
        f"battery mass is {battery.limiting}-dominated: "
        f"energy {battery.mass_energy_limited:.2f} kg, power "
        f"{battery.mass_power_limited:.2f} kg before structure margin",
    ))
    gates.append(InjectorGate(
        "battery_heat",
        "warn" if battery.heat > 0.10 * max(battery.electric_power, 1.0) else "pass",
        f"battery discharge heat {battery.heat:.0f} W",
    ))

    if any(g.status == "fail" for g in gates):
        suggestions.insert(0, _requirement_message())
    elif any(g.status == "warn" for g in gates):
        suggestions.append(
            "Pump sizing is preliminary; verify with pump curves, motor maps, "
            "battery pulse data, and cold-flow cavitation testing."
        )
    else:
        suggestions.append(
            "No screening gates failed; continue with real pump curves, motor "
            "thermal analysis, and cold-flow verification."
        )
    feasible = not any(g.status == "fail" for g in gates)
    return PumpFeasibility(feasible=feasible, gates=gates, suggestions=suggestions)


def size_electric_pumps(
    ledger: FeedSystemLedger,
    spec: PumpSizingSpec | None = None,
) -> ElectricPumpSizingResult:
    """Size electric pump feed hardware from an injector feed-system ledger."""
    spec = spec or PumpSizingSpec()
    lines: dict[str, PumpLineSizing] = {}
    total_electric = 0.0
    selected_bus_voltage = spec.drive.voltage or spec.battery.voltage or 0.0
    for role, ln in ledger.lines.items():
        eta, eta_source = _line_efficiency(
            spec, role, ln.volumetric_flow, ln.required_pump_head
        )
        hydraulic = None
        shaft = None
        rpm_source = None
        drive = None
        impeller = None
        inducer = None
        diffuser = None
        meanline = None
        curve = None
        if _finite(ln.required_pressure_rise):
            rise = max(0.0, float(ln.required_pressure_rise))
            hydraulic = ln.volumetric_flow * rise
            head = ln.required_pump_head if ln.required_pump_head is not None else 0.0
            rpm, rpm_source = _select_rpm(ln.volumetric_flow, head, spec)
            impeller = _impeller_geometry(role, ln.volumetric_flow, head, rpm, spec)
            meanline = _hydraulic_meanline(
                role, ln.volumetric_flow, head, ln, impeller, spec
            )
            if eta_source != "user":
                eta = meanline.hydraulic_efficiency
                eta_source = meanline.efficiency_source
            shaft = hydraulic / max(eta, 1e-9)
            drive = _drive_sizing(role, shaft, rpm, spec)
            total_electric += drive.electric_power
            selected_bus_voltage = max(selected_bus_voltage, drive.voltage)
            inducer = _inducer_geometry(role, ln.volumetric_flow, ln, impeller, spec)
            diffuser = _diffuser_volute_geometry(role, ln.volumetric_flow, impeller, spec)
            curve = _pump_performance_curve(
                role, ln.volumetric_flow, head, ln.density, rpm, eta
            )
        lines[role] = PumpLineSizing(
            role=role,
            pressure_rise=ln.required_pressure_rise,
            head=ln.required_pump_head,
            volumetric_flow=ln.volumetric_flow,
            hydraulic_power=hydraulic,
            shaft_power=shaft,
            efficiency=eta,
            efficiency_source=eta_source,
            rpm_source=rpm_source,
            drive=drive,
            impeller=impeller,
            inducer=inducer,
            diffuser_volute=diffuser,
            hydraulic_meanline=meanline,
            performance_curve=curve,
        )

    if selected_bus_voltage <= 0.0:
        selected_bus_voltage = 48.0
    battery = _battery_sizing(total_electric, spec, selected_bus_voltage)
    feasibility = _feasibility(ledger, lines, battery, spec)
    assumptions = {
        "burn_time_s": spec.burn_time,
        "pump_rpm": spec.drive.rpm if spec.drive.rpm is not None else "auto",
        "selected_bus_voltage_v": selected_bus_voltage,
        "motor_voltage_v": spec.drive.voltage if spec.drive.voltage is not None else "auto",
        "battery_voltage_v": spec.battery.voltage or spec.drive.voltage or selected_bus_voltage,
        "motor_efficiency": spec.drive.motor_efficiency,
        "inverter_efficiency": spec.drive.inverter_efficiency,
        "motor_power_density_w_per_kg": spec.drive.motor_power_density,
        "inverter_power_density_w_per_kg": spec.drive.inverter_power_density,
        "legacy_combined_motor_controller_power_density_w_per_kg": spec.drive.power_density,
        "battery_energy_density_j_per_kg": spec.battery.energy_density,
        "battery_power_density_w_per_kg": spec.battery.power_density,
        "battery_discharge_efficiency": spec.battery.discharge_efficiency,
        "battery_structural_margin": spec.battery.structural_margin,
        "head_coefficient": spec.head_coefficient,
        "flow_coefficient": spec.flow_coefficient,
        "material_tip_speed_limit_m_s": spec.material_tip_speed_limit,
        "pump_efficiency_by_role": spec.pump_efficiency,
        "pump_efficiency_model": "centrifugal meanline loss model unless pump_efficiency is supplied",
        "target_specific_speed": spec.target_specific_speed,
        "impeller_diameter_bounds_m": [
            spec.min_impeller_diameter,
            spec.max_impeller_diameter,
        ],
        "min_outlet_width_m": spec.min_outlet_width,
        "auto_current_target_a": spec.auto_current_target,
        "dependency_chain": [
            "engine/injector ledger sets mass flow, volumetric flow, required pump outlet pressure, tank/inlet pressure, and NPSH margin",
            "flow and pressure rise set hydraulic power: P_hyd = delta_p * Q",
            "centrifugal meanline velocity triangles and losses estimate pump efficiency",
            "pump efficiency converts hydraulic power to shaft power",
            "solver selects rpm from specific speed plus impeller diameter/outlet-width constraints unless rpm is supplied",
            "selected rpm converts shaft power to torque and sets impeller diameter through head coefficient",
            "motor and inverter efficiencies convert shaft power to electrical power, current, and heat",
            "battery energy density, power density, voltage, discharge efficiency, and burn time set pack mass/current/heat",
            "fluid density, vapor pressure/inlet pressure, rpm, and inlet geometry set cavitation/NPSH risk",
        ],
        "literature_sources": LITERATURE_SOURCES,
    }
    notes = [
        "Consumes FeedSystemLedger output; it does not re-solve injector/feed "
        "pressure requirements.",
        "Centrifugal/inducer/diffuser dimensions, velocity triangles, losses, "
        "and pump curves are preliminary meanline screening estimates for "
        "trade studies.",
        "Defaults are literature-backed screening values, not universal constants; "
        "pump efficiency and rpm are solved from duty unless supplied; replace "
        "technology defaults with actual pump curves, motor maps, inverter data, "
        "and battery pulse/thermal data as hardware is selected.",
    ]
    return ElectricPumpSizingResult(
        feasible=feasibility.feasible,
        lines=lines,
        battery=battery,
        feasibility=feasibility,
        assumptions=assumptions,
        notes=notes,
    )
