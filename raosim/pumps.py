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
    "rotor_material_density": 7800.0,
    "rotor_yield_strength": 600.0e6,
    "casing_yield_strength": 300.0e6,
    "structural_fos": 1.5,
    "bearing_dn_limit": 1.0e6,
    "seal_face_speed_limit": 35.0,
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
    # None = select the minimum blade number from the SP-8109 fig. 16
    # chart (digitized) and snap it to a multiple of the inducer blade
    # count (SP-8052 sec. 3.1.14); an int forces the count.
    blade_count: int | None = None
    # SP-8052 sec. 3.1.14: 2..5 blades, three preferred (odd count prevents
    # alternate cavitation).
    inducer_blade_count: int = 3
    # SP-8052 sec. 3.1.15: solidity 2.5 for a low-head inducer proper
    # (flat-plate inlet region 2.0..2.5; Hong et al. 2012 flew 2.6).
    inducer_solidity: float = 2.5
    inducer_hub_ratio: float = 0.35
    # SP-8052 sec. 3.1.9: incidence-to-blade-angle ratio alpha/beta is the
    # cavitation design variable; 0.35 (thin blades) .. 0.50 (thick),
    # "a mean value, 0.425, has gained preference".
    inducer_incidence_to_blade_ratio: float = 0.425
    diffuser_vane_count: int = 8
    material_tip_speed_limit: float = 350.0
    rotor_material_density: float = SCREENING_DEFAULTS["rotor_material_density"]
    rotor_yield_strength: float = SCREENING_DEFAULTS["rotor_yield_strength"]
    casing_yield_strength: float = SCREENING_DEFAULTS["casing_yield_strength"]
    structural_fos: float = SCREENING_DEFAULTS["structural_fos"]
    blade_thickness_ratio: float = 0.012
    # 0.4 mm manufacturing floor; small-pump performance is sensitive to
    # blade thickness/outlet angle at this scale (corpus: "Influence of
    # Blade Outlet Angle and Blade Thickness ... Mini Centrifugal Pump",
    # IOP mini-impeller studies) - treat thinner blades as unbuildable
    # rather than better.
    min_blade_thickness: float = 4.0e-4
    # Tapered leading edge; kept distinct from the discharge/root thickness.
    # This is a manufacturing-process input and is exported for inspection.
    min_impeller_leading_edge_thickness: float = 2.0e-4
    # SP-8109 inlet free-area practice is satisfied with only the full-length
    # main blades at the eye; the remaining discharge blades begin as
    # splitters downstream.  Four main blades is the default in the cited
    # 4..8 inlet-blade range.
    impeller_inlet_blade_count: int = 4
    splitter_start_radius_fraction: float = 0.55
    max_impeller_inlet_blockage: float = 0.20
    max_impeller_exit_blockage: float = 0.15
    shaft_diameter_ratio: float = 0.30
    min_shaft_diameter: float = 6.0e-3
    # These are hydraulic/mechanical interface dimensions, not CAD repair
    # constants.  The annular eye, inducer hub, and velocity triangle are
    # solved around them before a reference solid is constructed.
    shaft_fit_radial_clearance: float = 15.0e-6
    min_impeller_hub_wall_thickness: float = 3.0e-4
    min_inducer_hub_wall_thickness: float = 2.0e-4
    casing_wall_thickness_ratio: float = 0.035
    min_casing_wall_thickness: float = 1.5e-3
    split_casing_machining_tool_diameter: float = 5.0e-4
    split_casing_joint_separation_factor: float = 1.5
    # Wear-ring radial clearance is a manufacturing/vendor input, not a
    # meanline output; when provided, the SP-8109 sec. 3.5.2.1 rule
    # (balance-hole flow area ~= 4 x seal-clearance area) sizes the
    # impeller balance holes.
    wear_ring_radial_clearance: float | None = None
    bearing_dn_limit: float = SCREENING_DEFAULTS["bearing_dn_limit"]
    seal_face_speed_limit: float = SCREENING_DEFAULTS["seal_face_speed_limit"]
    seal_friction_coefficient: float = 0.04
    max_propellant_temperature_rise: float = 15.0
    fluid_specific_heat: dict[str, float] = field(default_factory=lambda: {
        "fuel": 2100.0,
        "oxidizer": 1700.0,
        "default": 2000.0,
    })
    max_head_per_stage: float = 2500.0
    target_specific_speed: float = 0.45
    min_impeller_diameter: float = 0.008
    max_impeller_diameter: float = 0.18
    min_outlet_width: float = 3.0e-4
    max_outlet_width_ratio: float = 0.12
    auto_current_target: float = SCREENING_DEFAULTS["auto_current_target"]
    # If neither --motor-voltage nor --battery-voltage is supplied, the default
    # architecture is one common DC pack/bus feeding both propellant pump drives.
    # Per-stream voltages would require explicit DC/DC converters or separate
    # packs, so the screening solver does not silently assume them.
    shared_bus: bool = True
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
    blade_count_source: str | None = None
    legacy_screening_inlet_angle_deg: float | None = None
    inlet_blade_count: int | None = None
    splitter_blade_count: int = 0
    splitter_start_radius_fraction: float | None = None
    blade_thickness: float | None = None
    blade_thickness_source: str | None = None
    blade_root_structural_minimum_thickness: float | None = None
    blade_root_structural_geometry_limited: bool = False
    inlet_blade_thickness: float | None = None
    inlet_blockage_fraction: float | None = None
    exit_blockage_fraction: float | None = None
    target_flow_coefficient: float | None = None

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
            "blade_count_source": self.blade_count_source,
            "inlet_blade_angle_deg": self.inlet_blade_angle_deg,
            "legacy_screening_inlet_angle_deg": (
                self.legacy_screening_inlet_angle_deg
            ),
            "inlet_blade_count": self.inlet_blade_count,
            "splitter_blade_count": self.splitter_blade_count,
            "splitter_start_radius_fraction": (
                self.splitter_start_radius_fraction
            ),
            "blade_thickness_m": self.blade_thickness,
            "blade_thickness_source": self.blade_thickness_source,
            "blade_root_structural_minimum_thickness_m": (
                self.blade_root_structural_minimum_thickness
            ),
            "blade_root_structural_geometry_limited": (
                self.blade_root_structural_geometry_limited
            ),
            "inlet_blade_thickness_m": self.inlet_blade_thickness,
            "inlet_blockage_fraction": self.inlet_blockage_fraction,
            "exit_blockage_fraction": self.exit_blockage_fraction,
            "target_flow_coefficient": self.target_flow_coefficient,
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
    # SP-8052 flat-plate inducer blade geometry (secs. 2.1.9/3.1.9,
    # 2.1.10/3.1.10): inlet tip blade angle from the tip flow coefficient
    # plus the incidence set by the alpha/beta design ratio; constant-lead
    # helix r*tan(beta) = const gives the hub angle and the pitch.
    inlet_flow_coefficient: float | None = None
    inlet_flow_angle_deg: float | None = None
    inlet_tip_blade_angle_deg: float | None = None
    hub_blade_angle_deg: float | None = None
    incidence_deg: float | None = None
    incidence_to_blade_ratio: float | None = None
    leading_edge_thickness: float | None = None

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
            "inlet_flow_coefficient": self.inlet_flow_coefficient,
            "inlet_flow_angle_deg": self.inlet_flow_angle_deg,
            "inlet_tip_blade_angle_deg": self.inlet_tip_blade_angle_deg,
            "hub_blade_angle_deg": self.hub_blade_angle_deg,
            "incidence_deg": self.incidence_deg,
            "incidence_to_blade_ratio": self.incidence_to_blade_ratio,
            "leading_edge_thickness_m": self.leading_edge_thickness,
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
    inlet_relative_flow_angle_deg: float
    inlet_blade_metal_angle_deg: float
    inlet_incidence_deg: float
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
            "inlet_relative_flow_angle_deg": (
                self.inlet_relative_flow_angle_deg
            ),
            "inlet_blade_metal_angle_deg": self.inlet_blade_metal_angle_deg,
            "inlet_incidence_deg": self.inlet_incidence_deg,
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
class PumpArchitectureClassification:
    role: str
    primary_type: str
    stage_mode: str
    suction_assist: str
    electric_architecture: str
    candidate_types: list[str]
    rationale: list[str]
    metrics: dict[str, float | int | str | None]
    source_ids: list[str]

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "primary_type": self.primary_type,
            "stage_mode": self.stage_mode,
            "suction_assist": self.suction_assist,
            "electric_architecture": self.electric_architecture,
            "candidate_types": self.candidate_types,
            "rationale": self.rationale,
            "metrics": self.metrics,
            "source_ids": self.source_ids,
        }


@dataclass
class PumpHardwareBOMItem:
    role: str
    subsystem: str
    component: str
    quantity: int
    status: str
    mass_estimate_kg: float | None
    key_parameters: dict[str, float | int | str | None]
    editable_reference_id: str | None
    source_ids: list[str]

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "subsystem": self.subsystem,
            "component": self.component,
            "quantity": self.quantity,
            "status": self.status,
            "mass_estimate_kg": self.mass_estimate_kg,
            "key_parameters": self.key_parameters,
            "editable_reference_id": self.editable_reference_id,
            "source_ids": self.source_ids,
        }


@dataclass
class PumpReferenceGeometry:
    role: str
    coordinate_system: str
    editable: bool
    source_ids: list[str]
    meridional_profile: list[dict[str, float | str]]
    impeller_disk: dict[str, float | int | str]
    blade_envelope: dict[str, float | int | str]
    inducer_helix: dict[str, float | int | str | None]
    diffuser_vane_ring: dict[str, float | int | str]
    volute_scroll: dict[str, float | int | str]
    shaft_datum: dict[str, float | str]
    ports: dict[str, dict[str, float | str]]
    notes: list[str]
    meridional_channel: dict | None = None
    thrust_balance: dict | None = None

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "coordinate_system": self.coordinate_system,
            "editable": self.editable,
            "source_ids": self.source_ids,
            "meridional_profile": self.meridional_profile,
            "impeller_disk": self.impeller_disk,
            "blade_envelope": self.blade_envelope,
            "inducer_helix": self.inducer_helix,
            "diffuser_vane_ring": self.diffuser_vane_ring,
            "volute_scroll": self.volute_scroll,
            "shaft_datum": self.shaft_datum,
            "ports": self.ports,
            "notes": self.notes,
            "meridional_channel": self.meridional_channel,
            "thrust_balance": self.thrust_balance,
        }


@dataclass
class PumpSystemCurvePoint:
    flow_ratio: float
    throttle: float
    volumetric_flow: float
    required_pressure_rise: float
    pump_pressure_rise: float
    pressure_margin: float
    status: str

    def to_dict(self) -> dict:
        return {
            "flow_ratio": self.flow_ratio,
            "throttle": self.throttle,
            "volumetric_flow_m3_s": self.volumetric_flow,
            "required_pressure_rise_pa": self.required_pressure_rise,
            "pump_pressure_rise_pa": self.pump_pressure_rise,
            "pressure_margin_pa": self.pressure_margin,
            "status": self.status,
        }


@dataclass
class PumpSystemCurve:
    role: str
    model: str
    source_ids: list[str]
    points: list[PumpSystemCurvePoint]
    supported_throttle_range: list[float] | None
    notes: list[str]

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "model": self.model,
            "source_ids": self.source_ids,
            "points": [p.to_dict() for p in self.points],
            "supported_throttle_range": self.supported_throttle_range,
            "notes": self.notes,
        }


@dataclass
class PumpThermalStressLedger:
    role: str
    source_ids: list[str]
    thermal: dict[str, float | str | None]
    stress: dict[str, float | str | None]
    loads: dict[str, float | str | None]
    margins: dict[str, float | None]

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "source_ids": self.source_ids,
            "thermal": self.thermal,
            "stress": self.stress,
            "loads": self.loads,
            "margins": self.margins,
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
    architecture: PumpArchitectureClassification | None = None
    reference_geometry: PumpReferenceGeometry | None = None
    system_curve: PumpSystemCurve | None = None
    thermal_stress: PumpThermalStressLedger | None = None

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
            "architecture": (
                self.architecture.to_dict() if self.architecture else None
            ),
            "reference_geometry": (
                self.reference_geometry.to_dict()
                if self.reference_geometry else None
            ),
            "system_curve": (
                self.system_curve.to_dict() if self.system_curve else None
            ),
            "thermal_stress": (
                self.thermal_stress.to_dict() if self.thermal_stress else None
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
    hardware_bom: list[PumpHardwareBOMItem] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "feasible": self.feasible,
            "lines": {k: v.to_dict() for k, v in self.lines.items()},
            "battery": self.battery.to_dict(),
            "feasibility": self.feasibility.to_dict(),
            "assumptions": self.assumptions,
            "hardware_bom": [item.to_dict() for item in self.hardware_bom],
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


def _margin_status(margin: float | None, *, warn: float = 1.25) -> str:
    if margin is None:
        return "pass"
    if margin < 1.0:
        return "fail"
    if margin < warn:
        return "warn"
    return "pass"


def _standard_bus_at_or_above(required_voltage: float) -> tuple[float, str]:
    for candidate in (24.0, 48.0, 96.0, 120.0, 270.0, 400.0, 540.0, 800.0):
        if candidate >= required_voltage:
            return candidate, "auto_standard_bus"
    return required_voltage, "auto_minimum_for_current_limit"


def _select_voltage(electric_power: float, spec: PumpSizingSpec) -> tuple[float, str]:
    drive = spec.drive
    if drive.voltage is not None:
        return float(drive.voltage), "user"
    if spec.battery.voltage is not None:
        return float(spec.battery.voltage), "battery_user"
    current_target = (
        drive.max_current
        or spec.battery.max_current
        or max(spec.auto_current_target, 1.0)
    )
    required = electric_power / max(current_target, 1e-9)
    return _standard_bus_at_or_above(required)


def _select_shared_bus_voltage(
    line_electric_power: dict[str, float],
    spec: PumpSizingSpec,
) -> tuple[float, str]:
    """Select one DC bus for every pump drive in the default architecture."""
    drive = spec.drive
    batt = spec.battery
    if batt.voltage is not None:
        return float(batt.voltage), "user_battery_shared_bus"
    if drive.voltage is not None:
        return float(drive.voltage), "user_motor_shared_bus"
    if not line_electric_power:
        return 48.0, "auto_default_shared_bus"

    per_drive_limit = drive.max_current
    pack_limit = batt.max_current
    current_target = max(spec.auto_current_target, 1.0)
    required = 0.0
    for power in line_electric_power.values():
        required = max(required, power / max(per_drive_limit or current_target, 1e-9))
    total_power = sum(max(0.0, p) for p in line_electric_power.values())
    if pack_limit is not None:
        required = max(required, total_power / max(pack_limit, 1e-9))
    elif per_drive_limit is None:
        # In the default no-hardware-map case, keep total pack current around
        # the same target as the per-drive current.  This avoids the old behavior
        # where two independently selected low-voltage drives were summarized as
        # one higher-voltage pack after the fact.
        required = max(required, total_power / current_target)
    voltage, source = _standard_bus_at_or_above(required)
    return voltage, f"shared_{source}"


def _drive_sizing(
    role: str,
    shaft_power: float,
    rpm: float,
    spec: PumpSizingSpec,
    *,
    bus_voltage: float | None = None,
    voltage_source: str | None = None,
) -> ElectricDriveSizing:
    drive = spec.drive
    eta_m = max(1e-6, min(1.0, drive.motor_efficiency))
    eta_i = max(1e-6, min(1.0, drive.inverter_efficiency))
    motor_input_power = shaft_power / eta_m
    electric_power = motor_input_power / eta_i
    omega = 2.0 * math.pi * max(rpm, 1e-9) / 60.0
    if bus_voltage is None:
        voltage, voltage_source = _select_voltage(electric_power, spec)
    else:
        voltage = float(bus_voltage)
        voltage_source = voltage_source or "shared_bus"
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
    voltage = (
        selected_bus_voltage
        if spec.shared_bus
        else (batt.voltage or spec.drive.voltage or selected_bus_voltage)
    )
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


# NASA SP-8109 fig. 16 (19740020848.pdf, printed p. 30): "minimum number
# of blades that can satisfy impeller velocity-gradient limits", plotted as
# pump head coefficient psi = gH/U2^2 vs impeller discharge flow
# coefficient phi2 = cm2/U2 at best efficiency; zero prewhirl, SHROUDED
# impellers, inlet/discharge tip diameter ratio delta = 0.65.  Digitized
# from the 300-dpi scan at the carpet nodes (same practice as the Rao
# theta-angle tables); read-off accuracy ~ +/-0.02 in psi.  Each row:
# (Z, ((phi2, psi_max), ...)) with phi2 ascending along the Z curve.
_SP8109_FIG16_MIN_BLADES: tuple[tuple[int, tuple[tuple[float, float], ...]], ...] = (
    (3, ((0.095, 0.350), (0.135, 0.310))),
    (4, ((0.085, 0.415), (0.125, 0.385), (0.165, 0.350), (0.205, 0.320),
         (0.250, 0.295))),
    (5, ((0.075, 0.465), (0.115, 0.440), (0.155, 0.410), (0.198, 0.378),
         (0.243, 0.350), (0.290, 0.320))),
    (6, ((0.070, 0.500), (0.110, 0.482), (0.150, 0.455), (0.192, 0.428),
         (0.236, 0.400), (0.283, 0.370))),
    (8, ((0.060, 0.555), (0.100, 0.542), (0.140, 0.520), (0.184, 0.498),
         (0.228, 0.474), (0.275, 0.448))),
    (10, ((0.055, 0.590), (0.095, 0.578), (0.135, 0.560), (0.178, 0.542),
          (0.221, 0.520), (0.268, 0.498), (0.298, 0.470))),
    (12, ((0.050, 0.625), (0.090, 0.612), (0.130, 0.598), (0.172, 0.582),
          (0.215, 0.562), (0.260, 0.542), (0.295, 0.510))),
    (16, ((0.042, 0.663), (0.122, 0.640), (0.205, 0.610), (0.250, 0.592),
          (0.290, 0.552))),
    (20, ((0.037, 0.688), (0.115, 0.668), (0.198, 0.640), (0.245, 0.622),
          (0.285, 0.578))),
    (24, ((0.078, 0.697), (0.200, 0.660), (0.270, 0.620))),
)


def _interp_clamped(nodes: tuple[tuple[float, float], ...], x: float) -> float:
    if x <= nodes[0][0]:
        return nodes[0][1]
    if x >= nodes[-1][0]:
        return nodes[-1][1]
    for (x0, y0), (x1, y1) in zip(nodes, nodes[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / max(x1 - x0, 1e-12)
    return nodes[-1][1]


def sp8109_min_blade_count(
    head_coefficient: float,
    discharge_flow_coefficient: float,
    *,
    multiple_of: int | None = None,
) -> dict:
    """Minimum impeller blade number from the SP-8109 fig. 16 chart.

    Returns the smallest digitized Z whose fig.-16 curve reaches the
    requested head coefficient at the given discharge flow coefficient
    (clamped to the charted phi2 range).  ``multiple_of`` optionally bumps
    the count to a multiple of the inducer blade number, per SP-8052
    sec. 3.1.14 ("whenever possible the blade number N be selected so that
    the impeller blade number is a multiple of N").
    """
    psi = float(head_coefficient)
    phi2 = float(discharge_flow_coefficient)
    selected = None
    for count, nodes in _SP8109_FIG16_MIN_BLADES:
        if _interp_clamped(nodes, phi2) >= psi:
            selected = count
            break
    if selected is None:
        selected = _SP8109_FIG16_MIN_BLADES[-1][0]
        status = "head_coefficient_beyond_digitized_chart"
    else:
        status = "within_chart"
    basis = "sp8109_fig16_min_blade_number"
    blade_count = selected
    if multiple_of is not None and multiple_of > 1:
        snapped = multiple_of * math.ceil(blade_count / multiple_of)
        if snapped != blade_count:
            basis += f"_snapped_to_multiple_of_{multiple_of}"
        blade_count = snapped
    return {
        "blade_count": blade_count,
        "chart_minimum": selected,
        "status": status,
        "basis": basis,
        "source": "NASA SP-8109 fig. 16 (zero prewhirl, shrouded, delta=0.65)",
    }


def _impeller_geometry(
    role: str,
    Q: float,
    head: float,
    rpm: float,
    spec: PumpSizingSpec,
    *,
    blade_root_structural_floor: float = 0.0,
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
    outlet_angle = 25.0 if stage_head > 0.0 else 0.0
    inlet_angle = math.degrees(math.atan2(phi, 1.0))
    if spec.blade_count is not None:
        blade_count = int(spec.blade_count)
        blade_count_source = "user_specified"
    else:
        chart = sp8109_min_blade_count(
            psi, phi, multiple_of=spec.inducer_blade_count
        )
        blade_count = chart["blade_count"]
        blade_count_source = chart["basis"]
        if chart["status"] != "within_chart":
            blade_count_source += f" ({chart['status']})"
    if blade_count <= 0:
        raise ValueError("impeller blade count must be positive")
    requested_inlet_count = min(
        blade_count, max(int(spec.impeller_inlet_blade_count), 1)
    )
    divisors = [
        count for count in range(4, min(8, blade_count) + 1)
        if blade_count % count == 0
    ]
    inlet_blade_count = (
        min(divisors, key=lambda count: (abs(count - requested_inlet_count), count))
        if divisors else blade_count
    )
    splitter_count = blade_count - inlet_blade_count

    # Enforce the SP-8109 free-area screen before fixing RPM.  With a
    # manufacturing-floor thickness, reducing RPM grows D2 and creates the
    # required circumferential passage.  A user-fixed RPM is rejected rather
    # than silently changed.
    beta2 = math.radians(max(outlet_angle, 1.0e-6))
    max_b2 = _clamp(float(spec.max_impeller_exit_blockage), 1e-3, 0.95)
    ratio_blockage = (
        blade_count * max(float(spec.blade_thickness_ratio), 0.0)
        / max(math.pi * math.sin(beta2), 1e-12)
    )
    if ratio_blockage > max_b2 + 1.0e-12:
        raise ValueError(
            "impeller blade-thickness ratio cannot satisfy the exit "
            f"free-area gate: blockage={ratio_blockage:.6g} > {max_b2:.6g}"
        )
    effective_blade_floor = max(
        spec.min_blade_thickness,
        float(blade_root_structural_floor),
    )
    d2_floor = (
        blade_count * effective_blade_floor
        / max(math.pi * math.sin(beta2) * max_b2, 1e-12)
    )
    structural_geometry_limited = False
    if d2 < d2_floor:
        if spec.drive.rpm is None:
            if d2_floor > spec.max_impeller_diameter:
                if blade_root_structural_floor <= 0.0:
                    raise ValueError(
                        "exit free-area closure requires an impeller larger "
                        "than max_impeller_diameter"
                    )
                d2 = spec.max_impeller_diameter
                structural_geometry_limited = True
            else:
                d2 = d2_floor
            omega = 2.0 * tip_speed / max(d2, 1e-12)
            rpm = omega * 60.0 / (2.0 * math.pi)
        elif blade_root_structural_floor > 0.0:
            structural_geometry_limited = True
    blade_t = max(
        effective_blade_floor,
        spec.blade_thickness_ratio * d2,
    )
    exit_blockage = (
        blade_count * blade_t
        / max(math.pi * d2 * math.sin(beta2), 1e-12)
    )
    open_exit_perimeter = math.pi * d2 * (1.0 - exit_blockage)
    if open_exit_perimeter <= 0.0:
        raise ValueError("impeller blades close the discharge flow area")
    required_b2 = Q / max(open_exit_perimeter * phi * tip_speed, 1e-12)
    b2 = max(required_b2, spec.min_outlet_width)
    if b2 > spec.max_outlet_width_ratio * d2 * (1.0 + 1e-9):
        if spec.drive.rpm is None:
            raise ValueError(
                "net-area discharge continuity requires an excessive impeller "
                "outlet width"
            )
    achieved_phi = Q / max(open_exit_perimeter * b2 * tip_speed, 1e-12)
    # Placeholder only; the coupled annular-eye solve below replaces D1.
    d1 = 0.45 * d2
    ns = omega * math.sqrt(max(Q, 0.0)) / max((G0 * stage_head) ** 0.75, 1e-12)
    ds = d2 * (G0 * stage_head) ** 0.25 / max(math.sqrt(max(Q, 1e-18)), 1e-12)
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
        flow_coefficient=achieved_phi,
        tip_speed=tip_speed,
        impeller_diameter=d2,
        inlet_diameter=d1,
        outlet_width=b2,
        blade_count=blade_count,
        inlet_blade_angle_deg=inlet_angle,
        outlet_blade_angle_deg=outlet_angle,
        recommendation=recommendation,
        blade_count_source=blade_count_source,
        legacy_screening_inlet_angle_deg=inlet_angle,
        inlet_blade_count=inlet_blade_count,
        splitter_blade_count=splitter_count,
        splitter_start_radius_fraction=_clamp(
            spec.splitter_start_radius_fraction, 0.05, 0.95
        ),
        blade_thickness=blade_t,
        blade_thickness_source=(
            "blade_root_structural_closure"
            if blade_root_structural_floor > max(
                spec.min_blade_thickness,
                spec.blade_thickness_ratio * d2,
            )
            else "manufacturing_or_diameter_ratio"
        ),
        blade_root_structural_minimum_thickness=(
            float(blade_root_structural_floor)
            if blade_root_structural_floor > 0.0 else None
        ),
        blade_root_structural_geometry_limited=structural_geometry_limited,
        inlet_blade_thickness=spec.min_impeller_leading_edge_thickness,
        exit_blockage_fraction=exit_blockage,
        target_flow_coefficient=phi,
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
    *,
    inlet_area: float | None = None,
) -> PumpVelocityTriangle:
    stages = max(impeller.stages, 1)
    stage_head = max(head, 0.0) / stages
    omega = 2.0 * math.pi * max(impeller.rpm, 1e-9) / 60.0
    u2 = max(impeller.tip_speed, 1e-9)
    u1 = omega * max(impeller.inlet_diameter, 0.0) / 2.0
    area1 = (
        float(inlet_area)
        if inlet_area is not None
        else math.pi * max(impeller.inlet_diameter, 1e-12) ** 2 / 4.0
    )
    if not math.isfinite(area1) or area1 <= 0.0:
        raise ValueError("pump velocity triangle requires positive inlet area")
    cm1 = max(Q, 0.0) / max(area1, 1e-12)
    cm2 = max(impeller.flow_coefficient * u2, 0.0)
    beta2 = math.radians(_clamp(impeller.outlet_blade_angle_deg, 5.0, 80.0))
    slip = _slip_factor(impeller.blade_count, impeller.outlet_blade_angle_deg)
    cu2 = max(0.0, slip * u2 - cm2 / max(math.tan(beta2), 1e-12))
    euler_head = u2 * cu2 / G0
    w1 = math.hypot(cm1, u1)
    w2 = math.hypot(cm2, max(u2 - cu2, 0.0))
    beta1_flow = math.degrees(
        math.atan2(cm1, max(u1 - 0.0, 1e-12))
    )
    # Zero prewhirl and zero design-point incidence are explicit assumptions;
    # the blade metal angle therefore equals the relative-flow angle.  Keep
    # ``inlet_blade_angle_deg`` as a backward-compatible alias.
    beta1_metal = beta1_flow
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
        inlet_blade_angle_deg=beta1_metal,
        inlet_relative_flow_angle_deg=beta1_flow,
        inlet_blade_metal_angle_deg=beta1_metal,
        inlet_incidence_deg=beta1_metal - beta1_flow,
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
    *,
    inlet_area: float | None = None,
) -> PumpHydraulicMeanline:
    triangle = _velocity_triangle(
        Q, head, impeller, spec, inlet_area=inlet_area
    )
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

    incidence_rad = math.radians(triangle.inlet_incidence_deg)
    incidence_loss = stage_head * (
        0.006 + min(0.20, 0.05 * (incidence_rad / 0.10) ** 2)
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
    # Keep the automatic meanline result conservative for small, high-head
    # rocket-pump screening.  Higher values should come from a real pump curve
    # or a user-supplied efficiency, not from this loss bucket model.
    eta = _clamp(stage_head / max(stage_head + total_loss, 1e-9), 0.20, 0.78)
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
        model="centrifugal_meanline_annular_eye_v2",
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


def _architecture_classification(
    role: str,
    ln,
    impeller: CentrifugalPumpGeometry,
    inducer: InducerGeometry,
    spec: PumpSizingSpec,
) -> PumpArchitectureClassification:
    ns = impeller.specific_speed
    stage_mode = "single_stage" if impeller.stages == 1 else "staged"
    suction_assist = (
        "inducer_required"
        if ln.npsh_margin is not None and ln.npsh_margin < 0.0 else
        "inducer_assisted" if inducer is not None else "none"
    )
    candidates: list[str] = []
    rationale: list[str] = []
    if ns < 0.20:
        primary = "radial_centrifugal_low_specific_speed"
        candidates.extend(["radial_centrifugal", "positive_displacement_candidate"])
        rationale.append("low nondimensional specific speed favors radial centrifugal duty; very low Ns can also justify a positive-displacement trade")
    elif ns <= 0.90:
        primary = "radial_centrifugal"
        candidates.append("radial_centrifugal")
        rationale.append("Ns lies in the radial centrifugal screening band used by SP-8109-style meanline sizing")
    elif ns <= 1.60:
        primary = "mixed_flow_candidate"
        candidates.extend(["mixed_flow", "radial_centrifugal_revisit"])
        rationale.append("high Ns pushes the current radial impeller toward mixed-flow geometry")
    else:
        primary = "axial_or_mixed_flow_candidate"
        candidates.extend(["axial", "mixed_flow", "multi_stage_centrifugal"])
        rationale.append("very high Ns is outside the radial centrifugal screening band")
    if impeller.stages > 1:
        candidates.append("staged_centrifugal")
        rationale.append(f"head exceeds the {spec.max_head_per_stage:.0f} m/stage screen")
    if ln.volumetric_flow < 1.0e-4 and (ln.required_pump_head or 0.0) > 1000.0:
        candidates.append("off_the_shelf_electric_feed_with_custom_validation")
        rationale.append("small high-head duty should be checked against miniature pump efficiency and cavitation data")
    candidates.append("electric_motor_driven")
    # Preserve order while deduping.
    candidates = list(dict.fromkeys(candidates))
    return PumpArchitectureClassification(
        role=role,
        primary_type=primary,
        stage_mode=stage_mode,
        suction_assist=suction_assist,
        electric_architecture="shared_pack_bus" if spec.shared_bus else "per_stream_drive_bus",
        candidate_types=candidates,
        rationale=rationale,
        metrics={
            "specific_speed": ns,
            "specific_diameter": impeller.specific_diameter,
            "flow_coefficient": impeller.flow_coefficient,
            "head_coefficient": impeller.head_coefficient,
            "stage_count": impeller.stages,
            "suction_specific_speed": inducer.suction_specific_speed,
            "npsh_margin_pa": inducer.npsh_margin,
        },
        source_ids=["NASA SP-8109", "NASA SP-8052", "Spiller, Stabile, Lentini 2013"],
    )


def _shaft_diameter(
    impeller: CentrifugalPumpGeometry,
    spec: PumpSizingSpec,
    *,
    shaft_power: float | None = None,
) -> float:
    """Return the larger packaging- and torsion-driven shaft diameter."""
    packaging = max(
        spec.min_shaft_diameter,
        spec.shaft_diameter_ratio * max(impeller.inlet_diameter, 0.35 * impeller.impeller_diameter),
    )
    if shaft_power is None or shaft_power <= 0.0:
        return packaging
    omega = 2.0 * math.pi * max(impeller.rpm, 1e-9) / 60.0
    torque = float(shaft_power) / omega
    shear_allow = (
        0.35 * spec.rotor_yield_strength
        / max(spec.structural_fos, 1e-9)
    )
    torsion = (
        16.0 * torque / max(math.pi * shear_allow, 1e-18)
    ) ** (1.0 / 3.0)
    return max(packaging, torsion)


def _solve_annular_eye_and_shaft(
    Q: float,
    impeller: CentrifugalPumpGeometry,
    spec: PumpSizingSpec,
    *,
    shaft_power: float | None = None,
) -> dict[str, float | int | bool | str]:
    """Couple eye diameter, shaft, root wall, and inlet flow coefficient.

    The solved radius satisfies

    ``Q = pi (R1^2 - Rh^2) phi1 omega R1``

    with ``Rh`` the larger of the requested inducer hub ratio and the shaft
    plus fit-clearance/root-wall envelope.  This removes the old full-disk
    continuity assumption and prevents CAD from shrinking the flow annulus.
    """
    q = max(float(Q), 0.0)
    omega = 2.0 * math.pi * max(impeller.rpm, 1e-9) / 60.0
    phi1 = max(float(spec.inlet_flow_coefficient), 1e-9)
    hub_ratio = _clamp(float(spec.inducer_hub_ratio), 0.0, 0.85)
    blade_t = float(
        impeller.inlet_blade_thickness or spec.min_blade_thickness
    )
    root_wall = max(spec.min_impeller_hub_wall_thickness, blade_t)
    inlet_blades = max(
        int(impeller.inlet_blade_count or impeller.blade_count), 1
    )
    beta_target = math.atan(phi1)

    def state(radius: float) -> tuple[float, float, float, float, float]:
        impeller.inlet_diameter = 2.0 * radius
        shaft_d = _shaft_diameter(
            impeller, spec, shaft_power=shaft_power
        )
        hub_r = max(
            hub_ratio * radius,
            0.5 * shaft_d + spec.shaft_fit_radial_clearance + root_wall,
        )
        area = math.pi * max(radius * radius - hub_r * hub_r, 0.0)
        blockage = (
            inlet_blades * blade_t
            / max(2.0 * math.pi * radius * math.sin(beta_target), 1e-18)
        )
        effective_area = area * max(1.0 - blockage, 0.0)
        capacity = effective_area * phi1 * omega * radius
        return capacity - q, shaft_d, hub_r, area, blockage

    lower = max(0.05 * impeller.impeller_diameter, 1.0e-6)
    # Find a sign-changing bracket.  The shaft packaging term grows only
    # linearly with R1 while capacity grows cubically for the allowed ratios.
    upper = max(lower * 1.05, lower + 1.0e-6)
    f_upper, _, _, _, _ = state(upper)
    expansions = 0
    while f_upper < 0.0 and expansions < 80:
        upper *= 1.20
        f_upper, _, _, _, _ = state(upper)
        expansions += 1
    if f_upper < 0.0:
        raise ValueError(
            "could not bracket a mechanically feasible annular pump eye"
        )
    f_lower, _, _, _, _ = state(lower)
    if f_lower >= 0.0:
        radius = lower
        iterations = 0
    else:
        iterations = 0
        for iterations in range(1, 101):
            radius = 0.5 * (lower + upper)
            f_mid, _, _, _, _ = state(radius)
            if abs(f_mid) <= max(q, 1.0e-12) * 1.0e-12:
                break
            if f_mid < 0.0:
                lower = radius
            else:
                upper = radius
        else:
            raise RuntimeError("annular pump-eye solve did not converge")
    residual, shaft_d, hub_r, area, blockage = state(radius)
    if hub_r >= radius:
        raise ValueError("shaft/hub envelope closes the pump inlet annulus")
    cm1 = q / max(area, 1e-18)
    effective_area = area * (1.0 - blockage)
    cm1_effective = q / max(effective_area, 1e-18)
    impeller.inlet_blockage_fraction = blockage
    return {
        "model": "coupled_annular_eye_shaft_bisection_v1",
        "converged": True,
        "iterations": iterations,
        "eye_radius_m": radius,
        "eye_hub_radius_m": hub_r,
        "inlet_area_m2": area,
        "effective_inlet_area_m2": effective_area,
        "inlet_meridional_velocity_m_s": cm1_effective,
        "gross_area_meridional_velocity_m_s": cm1,
        "inlet_flow_coefficient": (
            cm1_effective / max(omega * radius, 1e-18)
        ),
        "target_inlet_flow_coefficient": phi1,
        "inlet_blade_count": inlet_blades,
        "inlet_blockage_fraction": blockage,
        "inlet_free_area_fraction": 1.0 - blockage,
        "inlet_blockage_limit": spec.max_impeller_inlet_blockage,
        "inlet_blockage_status": (
            "pass" if blockage <= spec.max_impeller_inlet_blockage
            else "fail"
        ),
        "continuity_residual_m3_s": residual,
        "shaft_diameter_m": shaft_d,
        "shaft_fit_radial_clearance_m": spec.shaft_fit_radial_clearance,
        "impeller_hub_wall_thickness_m": root_wall,
        "shaft_power_coupled": shaft_power is not None,
    }


def _casing_wall_thickness(
    impeller: CentrifugalPumpGeometry,
    spec: PumpSizingSpec,
    *,
    design_pressure: float | None = None,
) -> float:
    empirical = max(
        spec.min_casing_wall_thickness,
        spec.casing_wall_thickness_ratio * impeller.impeller_diameter,
    )
    if design_pressure is None or design_pressure <= 0.0:
        return empirical
    radius = 0.68 * impeller.impeller_diameter
    allowable = spec.casing_yield_strength / max(spec.structural_fos, 1e-9)
    pressure_wall = float(design_pressure) * radius / max(allowable, 1e-9)
    return max(empirical, pressure_wall)


def _split_casing_joint_layout(
    casing_radius: float,
    casing_wall: float,
    volute_exit_area: float,
    design_pressure: float,
    spec: PumpSizingSpec,
) -> dict[str, float | int | str | bool]:
    """Preliminary pressure-loaded keyhole split-casing joint layout.

    The projected area includes the round scroll footprint plus the split
    tangential outlet neck.  This is an auditable clamp/land layout, not a
    gasket, preload-scatter, thread-engagement, or flange-flexibility
    qualification.
    """
    a_exit = math.sqrt(max(volute_exit_area, 0.0) / math.pi)
    wetted_radius = casing_radius + a_exit
    outlet_length = 3.0 * a_exit
    projected_area = (
        math.pi * wetted_radius**2 + 2.0 * a_exit * outlet_length
    )
    separation_factor = max(
        float(spec.split_casing_joint_separation_factor), 1.0
    )
    required_clamp = separation_factor * design_pressure * projected_area
    bolt_allow = spec.casing_yield_strength / max(spec.structural_fos, 1e-9)
    gasket_land = max(2.0e-3, 1.5 * casing_wall)

    # Clamp-driven bolt sizing must also FIT the flange: iterate the nominal
    # diameter up through standard coarse metric sizes until the required
    # count sits on the clear bolt-circle arc at no less than the minimum
    # pitch of 2.5 d (standard bolted-flange spacing practice; Shigley,
    # Mechanical Engineering Design — outside the propulsion corpus, generic
    # machine design).  Without this, a high-pressure/large-scroll casing
    # (e.g. the 13 kN LOX line: 56 x M3, evaluation 2026-07-13 FINDING 2)
    # requests more bolts than the flange circumference can carry and the
    # CAD layout correctly refuses.  The clear arc excludes the tangential
    # outlet-neck keep-out using the same criterion as
    # ``pump_cad_brep._split_joint_hole_layout``.
    min_pitch_factor = 2.5
    port_radius = a_exit  # outlet port area equals the scroll exit area
    bolt_d = None
    tensile_area = required_count = body_count = None
    edge_land = flange_outer_radius = bolt_radius = None
    clear_circumference = None
    fit_note = "first_size_fit"
    standard_sizes = [3.0e-3, 4.0e-3, 5.0e-3, 6.0e-3, 8.0e-3, 1.0e-2, 1.2e-2]
    start_d = max(3.0e-3, 0.08 * casing_radius)
    candidate_sizes = [d for d in standard_sizes if d >= start_d - 1e-12]
    if not candidate_sizes:
        candidate_sizes = [standard_sizes[-1]]
    for trial_d in candidate_sizes:
        trial_area = 0.75 * math.pi * trial_d**2 / 4.0
        trial_count = max(
            8,
            int(math.ceil(
                required_clamp / max(bolt_allow * trial_area, 1e-12)
            )),
        )
        trial_body = trial_count if trial_count % 2 == 0 else trial_count + 1
        trial_hole_r = 0.5 * 1.15 * trial_d
        trial_edge = max(1.5 * 1.15 * trial_d, 2.0e-3)
        trial_flange = wetted_radius + gasket_land + trial_edge
        trial_bolt_r = 0.5 * (
            wetted_radius + gasket_land + trial_flange - trial_hole_r
        )
        # Clear fraction of the bolt circle outside the outlet-neck keep-out
        # (segment x = casing_radius, y in [0, outlet_length], exclusion
        # radius port + gasket + 1.5 hole radii) — same rule as the CAD.
        exclusion = port_radius + gasket_land + 1.5 * trial_hole_r
        samples = 720
        clear = 0
        for k in range(samples):
            ang = 2.0 * math.pi * (k + 0.5) / samples
            x = trial_bolt_r * math.cos(ang)
            y = trial_bolt_r * math.sin(ang)
            y_near = min(max(y, 0.0), outlet_length)
            if math.hypot(x - casing_radius, y - y_near) > exclusion:
                clear += 1
        trial_clear_circ = 2.0 * math.pi * trial_bolt_r * clear / samples
        bolt_d = trial_d
        tensile_area = trial_area
        required_count = trial_count
        body_count = trial_body
        edge_land = trial_edge
        flange_outer_radius = trial_flange
        bolt_radius = trial_bolt_r
        clear_circumference = trial_clear_circ
        # One extra pitch of slack absorbs the CAD's discrete candidate
        # snapping and the clear-arc boundary at the outlet neck, so the
        # placed holes can never undercut the minimum pitch.
        if (trial_body + 1) * min_pitch_factor * trial_d <= trial_clear_circ:
            break
        fit_note = "diameter_grown_to_fit_clear_arc"
    bolt_layout_fits = (
        (body_count + 1) * min_pitch_factor * bolt_d <= clear_circumference
    )
    outlet_pair_count = 4
    total_count = body_count + outlet_pair_count
    flange_thickness = max(casing_wall, 1.5 * bolt_d)
    bolt_stress = required_clamp / max(total_count * tensile_area, 1e-12)
    min_scroll_section_d = 2.0 * a_exit / math.sqrt(24.0)
    return {
        "model": "keyhole_full_face_gasket_split_joint_v1",
        "parting_plane": "volute_scroll_centerplane",
        "design_pressure_pa": design_pressure,
        "projected_pressure_area_m2": projected_area,
        "pressure_separating_force_n": design_pressure * projected_area,
        "joint_separation_factor": separation_factor,
        "required_total_clamp_n": required_clamp,
        "body_bolt_count": body_count,
        "outlet_neck_bolt_count": outlet_pair_count,
        "total_bolt_count": total_count,
        "bolt_nominal_diameter_m": bolt_d,
        "bolt_hole_diameter_m": 1.15 * bolt_d,
        "bolt_min_pitch_m": min_pitch_factor * bolt_d,
        "bolt_min_pitch_rule": (
            "2.5d minimum bolt spacing (standard bolted-flange machine-design"
            " practice, Shigley; not a propulsion-corpus value)"
        ),
        "bolt_circle_radius_m": bolt_radius,
        "clear_bolt_circumference_m": clear_circumference,
        "bolt_layout_fits_clear_arc": bolt_layout_fits,
        "bolt_sizing_note": fit_note,
        "bolt_allowable_stress_pa": bolt_allow,
        "bolt_screen_stress_pa": bolt_stress,
        "bolt_screen_passed": bolt_stress <= bolt_allow,
        "gasket_selection": "full_face_keyhole_sheet_gasket_unqualified",
        "gasket_land_width_m": gasket_land,
        "minimum_free_edge_land_m": edge_land,
        "flange_outer_radius_m": flange_outer_radius,
        "flange_thickness_m": flange_thickness,
        "outlet_neck_length_m": outlet_length,
        "selected_scroll_tool_diameter_m": (
            spec.split_casing_machining_tool_diameter
        ),
        "minimum_modeled_scroll_section_diameter_m": min_scroll_section_d,
        "scroll_tool_access_passed": (
            spec.split_casing_machining_tool_diameter
            <= min_scroll_section_d
        ),
        "qualification": (
            "preliminary pressure/clamp/land layout only; gasket material, "
            "preload scatter, flange bending, fatigue, threads, dowel fits, "
            "thermal distortion, and proof testing remain required"
        ),
    }


def impeller_blade_camber(
    inlet_radius: float,
    outlet_radius: float,
    inlet_blade_angle_deg: float,
    outlet_blade_angle_deg: float,
    samples: int = 61,
) -> list[dict[str, float]]:
    """Blade camber line theta(r) between the solved velocity-triangle angles.

    Integrates d(theta)/dr = 1/(r*tan(beta(r))) with the blade angle beta
    varying linearly in radius from beta1 at the inlet eye to beta2 at the
    exit — the standard circular-arc/log-spiral centrifugal blade layout
    (NASA SP-8109 blade-geometry practice; constant beta recovers the pure
    log spiral).  Angles are measured from the tangential direction.
    Returns polar samples [{radius_m, theta_rad}] with theta(r1) = 0; wrap
    is positive, so CAD mirrors for rotation handedness.
    """
    r1 = float(inlet_radius)
    r2 = float(outlet_radius)
    beta1 = math.radians(float(inlet_blade_angle_deg))
    beta2 = math.radians(float(outlet_blade_angle_deg))
    if not 0.0 < r1 < r2:
        raise ValueError("camber line needs 0 < inlet radius < outlet radius")
    if min(beta1, beta2) <= 0.0 or max(beta1, beta2) >= 0.5 * math.pi:
        raise ValueError(
            "blade angles must lie strictly between 0 and 90 deg from "
            "tangential for a log-spiral-family camber line"
        )
    n = max(int(samples), 2)
    points: list[dict[str, float]] = []
    theta = 0.0
    prev_r = r1
    prev_slope = 1.0 / (r1 * math.tan(beta1))
    for i in range(n):
        r = r1 + (r2 - r1) * i / (n - 1)
        beta = beta1 + (beta2 - beta1) * (r - r1) / (r2 - r1)
        slope = 1.0 / (r * math.tan(beta))
        if i > 0:
            theta += 0.5 * (prev_slope + slope) * (r - prev_r)
        points.append({"radius_m": r, "theta_rad": theta})
        prev_r = r
        prev_slope = slope
    return points


def _meridional_channel(
    role: str,
    Q: float,
    impeller: CentrifugalPumpGeometry,
    inducer: InducerGeometry,
    axial_width: float,
    *,
    shaft_diameter: float | None = None,
    spec: PumpSizingSpec | None = None,
    eye_solve: dict | None = None,
    samples: int = 21,
) -> dict:
    """Quarter-ellipse hub/shroud meridional channel honoring D1, D2, b2.

    Replaces the coarse 6-station envelope with smooth hub and shroud
    curves from the axial eye annulus to the radial exit band, and screens
    the meridional-velocity progression against SP-8109 sec. 2.3.1.2
    ("the discharge meridional component of velocity c_m2 may vary from
    1 to 1.5 times the impeller inlet velocity").  Curve endpoints are
    exact: inlet area = pi*(r1^2 - r_hub^2), exit area = 2*pi*r2*b2.
    Coordinates: x axial (exit plane x = 0, flow arrives from -x), r radial.
    """
    r1 = 0.5 * impeller.inlet_diameter
    r2 = 0.5 * impeller.impeller_diameter
    b2 = impeller.outlet_width
    width = max(float(axial_width), 1.5 * b2)
    r_hub_raw = 0.5 * inducer.hub_ratio * inducer.diameter
    r_hub = r_hub_raw
    mechanical_hub = None
    if shaft_diameter is not None and spec is not None:
        blade_t = float(
            impeller.inlet_blade_thickness or spec.min_blade_thickness
        )
        wall = max(spec.min_impeller_hub_wall_thickness, blade_t)
        mechanical_hub = (
            0.5 * shaft_diameter
            + spec.shaft_fit_radial_clearance
            + wall
        )
        r_hub = max(r_hub, mechanical_hub)
    if eye_solve is not None:
        r_hub = max(r_hub, float(eye_solve["eye_hub_radius_m"]))
    if r_hub >= 0.95 * r1:
        raise ValueError(
            "coupled shaft/hub envelope leaves no credible impeller-eye annulus"
        )
    hub_adjusted = r_hub > r_hub_raw + 1.0e-12

    n = max(int(samples), 3)
    hub_curve: list[dict[str, float]] = []
    shroud_curve: list[dict[str, float]] = []
    areas: list[float] = []
    for i in range(n):
        theta = 0.5 * math.pi * i / (n - 1)
        s, c = math.sin(theta), math.cos(theta)
        x_h = -width * (1.0 - s)
        r_h = r2 - (r2 - r_hub) * c
        x_s = -width + (width - b2) * s
        r_s = r2 - (r2 - r1) * c
        hub_curve.append({"x_m": x_h, "r_m": r_h})
        shroud_curve.append({"x_m": x_s, "r_m": r_s})
        span = math.hypot(x_h - x_s, r_h - r_s)
        areas.append(2.0 * math.pi * 0.5 * (r_h + r_s) * span)

    q = max(float(Q), 0.0)
    inlet_area = areas[0]
    inlet_blockage = (
        float(eye_solve.get("inlet_blockage_fraction", 0.0))
        if eye_solve is not None else 0.0
    )
    effective_inlet_area = inlet_area * (1.0 - inlet_blockage)
    exit_area = areas[-1]
    exit_blockage = float(impeller.exit_blockage_fraction or 0.0)
    effective_exit_area = exit_area * (1.0 - exit_blockage)
    cm_inlet = q / max(effective_inlet_area, 1e-12)
    cm_exit = q / max(effective_exit_area, 1e-12)
    cm_ratio = cm_exit / max(cm_inlet, 1e-12)
    # SP-8109 sec. 2.3.1.2 discharge/inlet meridional-velocity practice.
    cm_status = "pass" if 1.0 <= cm_ratio <= 1.5 else "warn"
    effective_areas = [
        area * (
            1.0
            - inlet_blockage
            - (exit_blockage - inlet_blockage) * i / max(n - 1, 1)
        )
        for i, area in enumerate(areas)
    ]
    contracting = all(
        a1 <= a0 * (1.0 + 1e-9)
        for a0, a1 in zip(effective_areas, effective_areas[1:])
    )
    return {
        "reference_id": f"{role}.meridional_channel",
        "model": "quarter_ellipse_coupled_annular_eye_v2",
        "hub_curve": hub_curve,
        "shroud_curve": shroud_curve,
        "eye_hub_radius_m": r_hub,
        "eye_hub_radius_mechanically_constrained": hub_adjusted,
        "mechanical_minimum_hub_radius_m": mechanical_hub,
        "shaft_diameter_m": shaft_diameter,
        "shaft_fit_radial_clearance_m": (
            spec.shaft_fit_radial_clearance if spec is not None else None
        ),
        "impeller_hub_wall_thickness_m": (
            max(
                spec.min_impeller_hub_wall_thickness,
                float(
                    impeller.inlet_blade_thickness
                    or spec.min_blade_thickness
                ),
            )
            if spec is not None else None
        ),
        "eye_solve": dict(eye_solve) if eye_solve is not None else None,
        "inlet_area_m2": inlet_area,
        "effective_inlet_area_m2": effective_inlet_area,
        "inlet_blockage_fraction": inlet_blockage,
        "inlet_free_area_fraction": 1.0 - inlet_blockage,
        "exit_area_m2": exit_area,
        "effective_exit_area_m2": effective_exit_area,
        "exit_blockage_fraction": exit_blockage,
        "exit_free_area_fraction": 1.0 - exit_blockage,
        "effective_area_profile_m2": effective_areas,
        "inlet_meridional_velocity_m_s": cm_inlet,
        "exit_meridional_velocity_m_s": cm_exit,
        "cm_ratio": cm_ratio,
        "cm_ratio_status": cm_status,
        "area_progression_contracting": contracting,
        "source": (
            "NASA SP-8109 sec. 2.3.1.2 impeller design practice "
            "(cm2 = 1 to 1.5 x impeller inlet velocity)"
        ),
    }


def _thrust_balance_geometry(
    role: str,
    impeller: CentrifugalPumpGeometry,
    channel: dict,
    shaft_d: float,
    spec: PumpSizingSpec,
) -> dict:
    """Wear-ring / seal-land / balance-hole hooks (SP-8109 secs. 2.5.2/3.5.2).

    Wear rings are the recommended thrust-balance method (sec. 3.5.2.1:
    insensitive to cavitation and axial clearance, unlike balance ribs).
    The hub-side ring is laid out at the solved eye diameter D1 (neutral
    equal-diameter start; the relative ring diameters are the thrust-trim
    variable and stay editable).  Balance holes vent the back cavity to the
    impeller inlet (sec. 3.5.2.2); when a wear-ring radial clearance is
    supplied, the hole area follows sec. 3.5.2.1: leakage flow area
    approximately FOUR TIMES the seal-clearance area.  The shaft seal land
    records the solved face speed against the screening limit.
    """
    d_ring = impeller.inlet_diameter
    omega = 2.0 * math.pi * max(impeller.rpm, 0.0) / 60.0
    face_speed = omega * 0.5 * shaft_d
    clearance = spec.wear_ring_radial_clearance
    hole_count = impeller.blade_count
    holes: dict[str, float | int | str | None]
    if clearance is not None and clearance > 0.0:
        clearance_area = math.pi * d_ring * clearance
        hole_area_total = 4.0 * clearance_area
        hole_d = math.sqrt(4.0 * hole_area_total / (math.pi * hole_count))
        holes = {
            "status": "sized",
            "count": hole_count,
            "diameter_m": hole_d,
            "total_area_m2": hole_area_total,
            "seal_clearance_area_m2": clearance_area,
        }
    else:
        holes = {
            "status": "clearance_not_specified",
            "count": hole_count,
            "diameter_m": None,
            "total_area_m2": None,
            "seal_clearance_area_m2": None,
        }
    return {
        "reference_id": f"{role}.thrust_balance",
        "selection": "impeller_wear_rings",
        "selection_source": (
            "NASA SP-8109 sec. 3.5.2.1: wear rings recommended over balance "
            "ribs (not subject to cavitation/axial-clearance force changes)"
        ),
        "hub_wear_ring_diameter_m": d_ring,
        "front_wear_ring_diameter_m": d_ring,
        "wear_ring_diameters_note": (
            "equal-diameter neutral start; relative diameters are the "
            "axial-thrust trim variable (SP-8109 sec. 2.5.2.1) and remain "
            "editable"
        ),
        "wear_ring_radial_clearance_m": clearance,
        "balance_holes": holes,
        "balance_hole_rule": (
            "leakage flow area ~= 4 x seal-clearance area "
            "(NASA SP-8109 sec. 3.5.2.1)"
        ),
        "shaft_seal_land": {
            "diameter_m": shaft_d,
            "face_speed_m_s": face_speed,
            "face_speed_limit_m_s": spec.seal_face_speed_limit,
            "status": (
                "pass" if face_speed <= spec.seal_face_speed_limit
                else "warn"
            ),
        },
        "eye_hub_radius_m": channel.get("eye_hub_radius_m"),
    }


def _reference_geometry(
    role: str,
    ln,
    impeller: CentrifugalPumpGeometry,
    inducer: InducerGeometry,
    diffuser: DiffuserVoluteGeometry,
    spec: PumpSizingSpec,
    meanline: PumpHydraulicMeanline | None = None,
    *,
    channel: dict | None = None,
    shaft_diameter: float | None = None,
) -> PumpReferenceGeometry:
    d2 = impeller.impeller_diameter
    d1 = impeller.inlet_diameter
    b2 = impeller.outlet_width
    shaft_d = (
        float(shaft_diameter)
        if shaft_diameter is not None
        else _shaft_diameter(impeller, spec)
    )
    casing_r = 0.68 * d2
    casing_pressure = max(
        float(ln.required_outlet_pressure),
        float(ln.required_pressure_rise or 0.0),
    )
    casing_t = _casing_wall_thickness(
        impeller, spec, design_pressure=casing_pressure
    )
    length = max(1.8 * d2, inducer.diameter + d2)
    meridional = [
        {"station": "inlet_port", "x_m": -0.55 * length, "radius_m": 0.50 * d1},
        {"station": "inducer_leading_edge", "x_m": -0.35 * length, "radius_m": 0.50 * inducer.diameter},
        {"station": "impeller_eye", "x_m": -0.08 * length, "radius_m": 0.50 * d1},
        {"station": "impeller_exit", "x_m": 0.0, "radius_m": 0.50 * d2},
        {"station": "diffuser_exit", "x_m": 0.22 * length, "radius_m": 0.58 * d2},
        {"station": "volute_exit", "x_m": 0.40 * length, "radius_m": casing_r},
    ]
    if channel is None:
        channel = _meridional_channel(
            role,
            ln.volumetric_flow,
            impeller,
            inducer,
            0.08 * length,
            shaft_diameter=shaft_d,
            spec=spec,
        )
    thrust_balance = _thrust_balance_geometry(
        role, impeller, channel, shaft_d, spec
    )
    return PumpReferenceGeometry(
        role=role,
        coordinate_system="axisymmetric pump datum: x along shaft, radius from shaft centerline",
        editable=True,
        source_ids=["NASA SP-8109", "NASA SP-8052"],
        meridional_profile=meridional,
        impeller_disk={
            "reference_id": f"{role}.impeller_disk",
            "outer_diameter_m": d2,
            "eye_diameter_m": d1,
            "outlet_width_m": b2,
            "stage_count": impeller.stages,
            "tip_speed_m_s": impeller.tip_speed,
        },
        blade_envelope={
            "reference_id": f"{role}.impeller_blade_envelope",
            "blade_count": impeller.blade_count,
            "inlet_blade_count": impeller.inlet_blade_count,
            "splitter_blade_count": impeller.splitter_blade_count,
            "splitter_start_radius_fraction": (
                impeller.splitter_start_radius_fraction
            ),
            "inlet_angle_deg": (
                meanline.velocity_triangle.inlet_blade_metal_angle_deg
                if meanline is not None
                else impeller.inlet_blade_angle_deg
            ),
            "inlet_relative_flow_angle_deg": (
                meanline.velocity_triangle.inlet_relative_flow_angle_deg
                if meanline is not None else None
            ),
            "inlet_incidence_deg": (
                meanline.velocity_triangle.inlet_incidence_deg
                if meanline is not None else None
            ),
            "legacy_screening_inlet_angle_deg": (
                impeller.legacy_screening_inlet_angle_deg
            ),
            "outlet_angle_deg": impeller.outlet_blade_angle_deg,
            "estimated_blade_thickness_m": impeller.blade_thickness,
            "inlet_blade_thickness_m": impeller.inlet_blade_thickness,
            "inlet_blockage_fraction": impeller.inlet_blockage_fraction,
            "exit_blockage_fraction": impeller.exit_blockage_fraction,
            "inlet_blockage_limit": spec.max_impeller_inlet_blockage,
            "exit_blockage_limit": spec.max_impeller_exit_blockage,
            "camber_handedness": (
                "positive theta is counter-clockwise viewed from +shaft; "
                "rotation handedness must be selected before release"
            ),
        },
        inducer_helix={
            "reference_id": f"{role}.inducer_helix",
            "diameter_m": inducer.diameter,
            "hub_ratio": inducer.hub_ratio,
            "blade_count": inducer.blade_count,
            "pitch_m": inducer.pitch,
            "wrap_angle_deg": inducer.wrap_angle_deg,
            # SP-8052 flat-plate blade geometry (secs. 3.1.9/3.1.10).
            "inlet_tip_blade_angle_deg": inducer.inlet_tip_blade_angle_deg,
            "hub_blade_angle_deg": inducer.hub_blade_angle_deg,
            "incidence_deg": inducer.incidence_deg,
            "inlet_flow_coefficient": inducer.inlet_flow_coefficient,
            "leading_edge_thickness_m": inducer.leading_edge_thickness,
            "shaft_fit_radial_clearance_m": (
                spec.shaft_fit_radial_clearance
            ),
            "hub_wall_thickness_m": max(
                spec.min_inducer_hub_wall_thickness,
                inducer.leading_edge_thickness or 0.0,
            ),
        },
        diffuser_vane_ring={
            "reference_id": f"{role}.diffuser_vane_ring",
            "selection": diffuser.selection,
            "vane_count": diffuser.vane_count,
            "vane_width_m": diffuser.vane_width,
            "throat_area_m2": diffuser.throat_area,
            # Solved absolute flow angle at impeller exit (atan cm2/cu2 with
            # slip): the vane inlet is set to the flow, SP-8109 diffusion
            # system practice.
            "vane_angle_deg": (
                meanline.velocity_triangle.outlet_absolute_flow_angle_deg
                if meanline is not None else None
            ),
        },
        volute_scroll={
            "reference_id": f"{role}.volute_scroll",
            "exit_area_m2": diffuser.volute_exit_area,
            "area_schedule": (
                "A(theta)=A_exit*theta/(2*pi); "
                "constant_mean_velocity_screen"
            ),
            "diffusion_ratio": diffuser.diffusion_ratio,
            "casing_inner_radius_m": casing_r,
            "casing_wall_thickness_m": casing_t,
            "design_pressure_pa": casing_pressure,
            "wall_sizing_model": (
                "max(manufacturing_floor, diameter_ratio, "
                "thin_wall_pressure_screen)"
            ),
            "split_casing_joint": _split_casing_joint_layout(
                casing_r,
                casing_t,
                diffuser.volute_exit_area,
                casing_pressure,
                spec,
            ),
        },
        shaft_datum={
            "reference_id": f"{role}.shaft_datum",
            "diameter_m": shaft_d,
            "estimated_span_m": length,
        },
        ports={
            "inlet": {
                "reference_id": f"{role}.inlet_port",
                "diameter_m": d1,
                "area_m2": math.pi * d1**2 / 4.0,
            },
            "outlet": {
                "reference_id": f"{role}.outlet_port",
                "area_m2": diffuser.volute_exit_area,
                "equivalent_diameter_m": _safe_sqrt(4.0 * diffuser.volute_exit_area / math.pi),
            },
        },
        notes=[
            "Reference geometry is an editable sizing envelope, not blade-resolved CAD.",
            "Use the reference IDs to map BOM rows to future CAD features.",
        ],
        meridional_channel=channel,
        thrust_balance=thrust_balance,
    )


def _system_curve_coupling(
    role: str,
    ln,
    curve: PumpPerformanceCurve,
) -> PumpSystemCurve:
    if ln.required_pressure_rise is None:
        return PumpSystemCurve(
            role=role,
            model="not_evaluated_missing_tank_pressure",
            source_ids=["Huzel and Huang SP-125", "NASA SP-8109"],
            points=[],
            supported_throttle_range=None,
            notes=["Tank/inlet pressure is needed before pump curve and system curve can be intersected."],
        )
    tank_pressure = ln.required_outlet_pressure - ln.required_pressure_rise
    dynamic_design = (
        ln.injector_dp + ln.manifold_loss + ln.regen_loss + ln.line_valve_loss
    )
    supported: list[float] = []
    points: list[PumpSystemCurvePoint] = []
    for p in curve.points:
        q = p.flow_ratio
        throttle = q
        required_outlet = (
            ln.chamber_pressure * throttle
            + dynamic_design * q * q
            + ln.control_margin * throttle
        )
        required_rise = max(0.0, required_outlet - tank_pressure)
        margin = p.pressure_rise - required_rise
        denom = max(required_rise, 1.0)
        if margin >= 0.0:
            status = "pass"
            supported.append(throttle)
        elif margin >= -0.10 * denom:
            status = "warn"
        else:
            status = "fail"
        points.append(PumpSystemCurvePoint(
            flow_ratio=q,
            throttle=throttle,
            volumetric_flow=p.volumetric_flow,
            required_pressure_rise=required_rise,
            pump_pressure_rise=p.pressure_rise,
            pressure_margin=margin,
            status=status,
        ))
    throttle_range = [min(supported), max(supported)] if supported else None
    return PumpSystemCurve(
        role=role,
        model="fixed_speed_pump_curve_vs_quadratic_feed_system_curve_v1",
        source_ids=["Huzel and Huang SP-125", "NASA SP-8109"],
        points=points,
        supported_throttle_range=throttle_range,
        notes=[
            "Chamber pressure is scaled linearly with throttle; injector, line, manifold, and regen losses are scaled with flow squared.",
            "Use a measured pump map and valve/sleeve schedule before treating the range as qualified.",
        ],
    )


# SP-8052 sec. 2.1.6: knife-sharp leading edge ideal; J-2/F-1 practice
# leaves the edge 0.005 to 0.010 in. thick.  Small pumps take the low end.
_INDUCER_LEADING_EDGE_THICKNESS = 0.005 * 0.0254


def _inducer_geometry(
    role: str,
    Q: float,
    ln,
    impeller: CentrifugalPumpGeometry,
    spec: PumpSizingSpec,
    *,
    shaft_diameter: float | None = None,
) -> InducerGeometry:
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
    blade_count = max(spec.inducer_blade_count, 1)
    hub_ratio = min(max(spec.inducer_hub_ratio, 0.0), 0.85)
    if shaft_diameter is not None:
        required_hub_r = (
            0.5 * shaft_diameter
            + spec.shaft_fit_radial_clearance
            + max(
                spec.min_inducer_hub_wall_thickness,
                _INDUCER_LEADING_EDGE_THICKNESS,
            )
        )
        hub_ratio = max(hub_ratio, required_hub_r / max(0.5 * eye_d, 1e-12))
    if hub_ratio >= 0.95:
        raise ValueError(
            "coupled shaft/root-wall envelope closes the inducer inlet annulus"
        )

    # SP-8052 secs. 2.1.9/3.1.9: the tip flow coefficient (zero prewhirl)
    # sets the flow angle; the blade angle carries it plus the incidence
    # from the alpha/beta design ratio (0.35 thin .. 0.50 thick, 0.425
    # preferred), the cavitation design variable.
    eye_area = 0.25 * math.pi * eye_d**2 * max(1.0 - hub_ratio**2, 1e-9)
    cm1 = max(Q, 0.0) / max(eye_area, 1e-12)
    tip_speed = omega * 0.5 * eye_d
    phi_tip = cm1 / max(tip_speed, 1e-9)
    flow_angle = math.atan(phi_tip)
    ratio = min(max(float(spec.inducer_incidence_to_blade_ratio), 0.0), 0.9)
    blade_angle = flow_angle / max(1.0 - ratio, 1e-9)
    incidence = blade_angle - flow_angle

    # SP-8052 secs. 2.1.10/3.1.10: flat-plate constant-lead helix,
    # lambda = r*tan(beta) constant, lead = 2*pi*lambda; the hub blade
    # angle follows from the same lead.
    pitch = 2.0 * math.pi * (0.5 * eye_d) * math.tan(blade_angle)
    hub_angle = math.atan(math.tan(blade_angle) / max(hub_ratio, 1e-9))
    # SP-8052 sec. 3.1.15 solidity with the developed helical chord
    # c = r*wrap/cos(beta): wrap = 2*pi*sigma*cos(beta)/N.
    wrap = math.degrees(
        2.0 * math.pi * spec.inducer_solidity * math.cos(blade_angle)
        / blade_count
    )

    recommendation = "axial inducer ahead of impeller eye"
    if ln.npsh_margin is not None and ln.npsh_margin < 0.0:
        recommendation = "negative NPSH margin; raise tank pressure/subcooling or lower inlet losses"
    elif suction_ns is not None and suction_ns > 4.0:
        recommendation = "high suction specific speed; reduce rpm or add inlet boost/inducer margin"
    return InducerGeometry(
        role=role,
        diameter=eye_d,
        hub_ratio=hub_ratio,
        blade_count=blade_count,
        solidity=spec.inducer_solidity,
        pitch=pitch,
        wrap_angle_deg=wrap,
        suction_specific_speed=suction_ns,
        npsh_margin=ln.npsh_margin,
        recommendation=recommendation,
        inlet_flow_coefficient=phi_tip,
        inlet_flow_angle_deg=math.degrees(flow_angle),
        inlet_tip_blade_angle_deg=math.degrees(blade_angle),
        hub_blade_angle_deg=math.degrees(hub_angle),
        incidence_deg=math.degrees(incidence),
        incidence_to_blade_ratio=ratio,
        leading_edge_thickness=_INDUCER_LEADING_EDGE_THICKNESS,
    )


def _diffuser_volute_geometry(role: str, Q: float, impeller: CentrifugalPumpGeometry, spec: PumpSizingSpec) -> DiffuserVoluteGeometry:
    cm2 = max(impeller.flow_coefficient * impeller.tip_speed, 1e-9)
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


def _margin(allowable: float, demand: float) -> float | None:
    if demand <= 0.0:
        return None
    return allowable / demand


def _thermal_stress_ledger(
    role: str,
    ln,
    sizing: PumpLineSizing,
    spec: PumpSizingSpec,
) -> PumpThermalStressLedger | None:
    if (
        sizing.impeller is None
        or sizing.inducer is None
        or sizing.drive is None
        or sizing.hydraulic_meanline is None
        or sizing.hydraulic_power is None
        or sizing.shaft_power is None
    ):
        return None
    imp = sizing.impeller
    ind = sizing.inducer
    drv = sizing.drive
    meanline = sizing.hydraulic_meanline
    tri = meanline.velocity_triangle
    rho_l = max(float(ln.density), 1e-9)
    mdot = max(ln.volumetric_flow * rho_l, 1e-12)
    cp = spec.fluid_specific_heat.get(role, spec.fluid_specific_heat.get("default", 2000.0))
    pump_loss_heat = max(0.0, sizing.shaft_power - sizing.hydraulic_power)
    disk_heat = rho_l * G0 * ln.volumetric_flow * meanline.losses.disk_friction_loss_head
    fluid_heat = 0.70 * pump_loss_heat + disk_heat
    fluid_delta_t = fluid_heat / max(mdot * cp, 1e-12)
    motor_heat_fraction = drv.total_heat / max(drv.shaft_power, 1.0)

    omega = 2.0 * math.pi * drv.rpm / 60.0
    shaft_d = float(
        (sizing.reference_geometry.shaft_datum or {}).get(
            "diameter_m", _shaft_diameter(
                imp, spec, shaft_power=sizing.shaft_power
            )
        )
    )
    casing_radius = 0.68 * imp.impeller_diameter
    rotor_allow = spec.rotor_yield_strength / max(spec.structural_fos, 1e-9)
    casing_allow = spec.casing_yield_strength / max(spec.structural_fos, 1e-9)
    shear_allow = 0.35 * spec.rotor_yield_strength / max(spec.structural_fos, 1e-9)

    impeller_hoop = spec.rotor_material_density * imp.tip_speed**2
    inducer_tip_speed = omega * ind.diameter / 2.0
    inducer_hoop = spec.rotor_material_density * inducer_tip_speed**2
    blade_force = rho_l * ln.volumetric_flow * abs(tri.outlet_whirl_velocity)
    force_per_blade = blade_force / max(imp.blade_count, 1)
    blade_thickness = max(
        float(imp.blade_thickness or 0.0),
        spec.min_blade_thickness,
        spec.blade_thickness_ratio * imp.impeller_diameter,
    )
    blade_section_modulus = max(imp.outlet_width * blade_thickness**2 / 6.0, 1e-15)
    blade_bending = force_per_blade * max(imp.outlet_width, 1e-6) / blade_section_modulus
    shaft_torsion = 16.0 * drv.torque / max(math.pi * shaft_d**3, 1e-18)
    casing_pressure = max(float(ln.required_outlet_pressure), float(ln.required_pressure_rise or 0.0))
    casing_t = _casing_wall_thickness(
        imp, spec, design_pressure=casing_pressure
    )
    casing_hoop = casing_pressure * casing_radius / max(casing_t, 1e-12)
    eye_area = math.pi * imp.inlet_diameter**2 / 4.0
    axial_thrust = max(float(ln.required_pressure_rise or 0.0), 0.0) * eye_area
    radial_load = 0.05 * max(float(ln.required_pressure_rise or 0.0), 0.0) * imp.impeller_diameter * max(imp.outlet_width, 1e-9)
    bearing_dn = shaft_d * 1000.0 * drv.rpm
    seal_face_speed = math.pi * shaft_d * drv.rpm / 60.0
    seal_heat = spec.seal_friction_coefficient * axial_thrust * seal_face_speed

    margins = {
        "motor_heat_fraction_to_50pct": _margin(0.50, motor_heat_fraction),
        "propellant_delta_t": _margin(spec.max_propellant_temperature_rise, fluid_delta_t),
        "impeller_rotating_stress": _margin(rotor_allow, impeller_hoop),
        "inducer_rotating_stress": _margin(rotor_allow, inducer_hoop),
        "blade_root_bending": _margin(rotor_allow, blade_bending),
        "shaft_torsion": _margin(shear_allow, shaft_torsion),
        "casing_pressure": _margin(casing_allow, casing_hoop),
        "bearing_dn": _margin(spec.bearing_dn_limit, bearing_dn),
        "seal_face_speed": _margin(spec.seal_face_speed_limit, seal_face_speed),
    }
    return PumpThermalStressLedger(
        role=role,
        source_ids=["NASA SP-8109", "NASA SP-8052", "NASA SP-8107/SP-8112 family"],
        thermal={
            "motor_heat_w": drv.motor_heat,
            "controller_heat_w": drv.controller_heat,
            "drive_total_heat_w": drv.total_heat,
            "drive_heat_fraction_of_shaft_power": motor_heat_fraction,
            "pump_loss_heat_w": pump_loss_heat,
            "disk_friction_heat_w": disk_heat,
            "estimated_propellant_heat_pickup_w": fluid_heat,
            "fluid_specific_heat_j_kg_k": cp,
            "estimated_propellant_temperature_rise_k": fluid_delta_t,
            "model": "loss_heat_to_fluid_screen_v1",
        },
        stress={
            "rotor_material_density_kg_m3": spec.rotor_material_density,
            "rotor_yield_strength_pa": spec.rotor_yield_strength,
            "rotor_allowable_pa": rotor_allow,
            "casing_yield_strength_pa": spec.casing_yield_strength,
            "casing_allowable_pa": casing_allow,
            "impeller_rotating_hoop_stress_pa": impeller_hoop,
            "inducer_rotating_hoop_stress_pa": inducer_hoop,
            "blade_root_bending_stress_pa": blade_bending,
            "blade_root_thickness_m": blade_thickness,
            "blade_root_thickness_source": imp.blade_thickness_source,
            "blade_root_structural_minimum_thickness_m": (
                imp.blade_root_structural_minimum_thickness
            ),
            "shaft_torsional_shear_pa": shaft_torsion,
            "casing_hoop_stress_pa": casing_hoop,
            "model": "screening_rotor_shaft_casing_bearing_seal_v1",
        },
        loads={
            "shaft_diameter_m": shaft_d,
            "casing_inner_radius_m": casing_radius,
            "casing_wall_thickness_m": casing_t,
            "casing_pressure_pa": casing_pressure,
            "axial_thrust_n": axial_thrust,
            "radial_load_n": radial_load,
            "bearing_dn_mm_rpm": bearing_dn,
            "bearing_dn_limit_mm_rpm": spec.bearing_dn_limit,
            "seal_face_speed_m_s": seal_face_speed,
            "seal_face_speed_limit_m_s": spec.seal_face_speed_limit,
            "seal_friction_heat_w": seal_heat,
        },
        margins=margins,
    )


def _estimate_component_masses(
    line: PumpLineSizing,
    spec: PumpSizingSpec,
) -> dict[str, float | None]:
    imp = line.impeller
    ind = line.inducer
    if imp is None or ind is None:
        return {}
    shaft_d = float(
        line.reference_geometry.shaft_datum["diameter_m"]
        if line.reference_geometry is not None
        else _shaft_diameter(imp, spec, shaft_power=line.shaft_power)
    )
    length = max(1.8 * imp.impeller_diameter, ind.diameter + imp.impeller_diameter)
    casing_r = 0.68 * imp.impeller_diameter
    casing_t = float(
        line.reference_geometry.volute_scroll["casing_wall_thickness_m"]
        if line.reference_geometry is not None
        else _casing_wall_thickness(imp, spec)
    )
    rho = spec.rotor_material_density
    impeller = rho * math.pi * (imp.impeller_diameter / 2.0) ** 2 * max(imp.outlet_width, 1e-6) * 0.55
    inducer = rho * math.pi * (ind.diameter / 2.0) ** 2 * max(0.35 * ind.diameter, 1e-6) * 0.20
    shaft = rho * math.pi * shaft_d**2 / 4.0 * length
    casing = rho * 4.0 * math.pi * casing_r**2 * casing_t * 0.55

    # ---- diffusion system ------------------------------------------------ #
    # Previously ``None``, which made ``_pump_hardware_mass`` withhold the
    # whole pump mass and left the engine dry-mass ledger permanently open.
    # The diffusion system's distinct metal is the vane ring standing between
    # the impeller tip and the casing wall: ``n_v`` radial vanes, each spanning
    # the radial gap, of axial extent ``vane_width``.  NASA SP-8109 (*Liquid
    # Rocket Engine Centrifugal Flow Turbopumps*, 1973,
    # ``propulsion_texts/fuel_pump_design/19740020848.pdf``) treats the
    # diffusion system as vaned diffuser / vaneless volute hardware carried by
    # the same casing; there is no published vane-thickness correlation, so the
    # vane is taken at the casing wall thickness -- both are machined from the
    # same pressure-boundary stock and sized by the same discharge pressure.
    # A vaneless volute has no vane ring and therefore contributes no extra
    # metal beyond the casing already counted above; that is an exact zero, not
    # a missing estimate.
    dif = line.diffuser_volute
    if dif is None:
        diffuser = None
    else:
        vane_count = int(getattr(dif, "vane_count", 0) or 0)
        vane_width = max(float(getattr(dif, "vane_width", 0.0) or 0.0), 0.0)
        radial_gap = max(casing_r - 0.5 * imp.impeller_diameter, 0.0)
        diffuser = rho * vane_count * vane_width * radial_gap * casing_t

    # ---- inlet / outlet port stubs --------------------------------------- #
    # Thin-wall stubs at the casing wall thickness, taken two diameters long
    # (the shortest run that still admits a weld/flange land and a straight
    # settling length upstream of the inducer).  Shell mass follows the same
    # relation as NASA SP-125 eq. 8-32: mid-surface circumference x wall x
    # length x density.
    outlet_d = _safe_sqrt(4.0 * float(dif.volute_exit_area) / math.pi) if dif else 0.0
    ports = 0.0
    for diameter in (float(imp.inlet_diameter), float(outlet_d)):
        if diameter <= 0.0:
            continue
        ports += (
            rho * math.pi * (diameter + casing_t) * casing_t * (2.0 * diameter)
        )

    # The drive masses are already solved by ``_drive_sizing`` from the
    # published power densities; surfacing them here means the BOM and the
    # ``ElectricDriveSizing`` rollup can never disagree.
    drv = line.drive
    return {
        "impeller": impeller,
        "inducer": inducer,
        "shaft_coupling": shaft,
        "casing": casing,
        "bearings": 0.12 * shaft,
        "seals": 0.08 * shaft,
        "diffuser_volute": diffuser,
        "ports": ports,
        "ports_instrumentation": ports,
        "motor": float(drv.motor_mass) if drv is not None else None,
        "inverter": float(drv.inverter_mass) if drv is not None else None,
    }


def _hardware_bom(
    lines: dict[str, PumpLineSizing],
    battery: BatterySizing,
    spec: PumpSizingSpec,
) -> list[PumpHardwareBOMItem]:
    items: list[PumpHardwareBOMItem] = []
    for role, line in lines.items():
        imp = line.impeller
        ind = line.inducer
        dif = line.diffuser_volute
        drv = line.drive
        if imp is None or ind is None or dif is None or drv is None:
            continue
        masses = _estimate_component_masses(line, spec)

        def add(
            subsystem: str,
            component: str,
            params: dict[str, float | int | str | None],
            *,
            mass_key: str | None = None,
            reference_id: str | None = None,
            status: str = "screening_sized",
            source_ids: list[str] | None = None,
        ) -> None:
            items.append(PumpHardwareBOMItem(
                role=role,
                subsystem=subsystem,
                component=component,
                quantity=1,
                status=status,
                mass_estimate_kg=(masses.get(mass_key) if mass_key else None),
                key_parameters=params,
                editable_reference_id=reference_id,
                source_ids=source_ids or ["NASA SP-8109"],
            ))

        add("hydraulic", "axial inducer", {
            "diameter_m": ind.diameter,
            "blade_count": ind.blade_count,
            "solidity": ind.solidity,
            "suction_specific_speed": ind.suction_specific_speed,
        }, mass_key="inducer", reference_id=f"{role}.inducer_helix",
           source_ids=["NASA SP-8052"])
        add("hydraulic", "centrifugal impeller", {
            "outer_diameter_m": imp.impeller_diameter,
            "outlet_width_m": imp.outlet_width,
            "blade_count": imp.blade_count,
            "rpm": imp.rpm,
            "stage_count": imp.stages,
        }, mass_key="impeller", reference_id=f"{role}.impeller_disk")
        add("hydraulic", dif.selection, {
            "throat_area_m2": dif.throat_area,
            "vane_count": dif.vane_count,
            "vane_width_m": dif.vane_width,
            "volute_exit_area_m2": dif.volute_exit_area,
        }, mass_key="diffuser_volute",
           reference_id=f"{role}.diffuser_vane_ring",
           source_ids=["NASA SP-8109"])
        add("mechanical", "shaft and coupling", {
            "shaft_diameter_m": (
                line.reference_geometry.shaft_datum["diameter_m"]
                if line.reference_geometry is not None
                else _shaft_diameter(imp, spec, shaft_power=line.shaft_power)
            ),
            "torque_n_m": drv.torque,
            "rpm": drv.rpm,
        }, mass_key="shaft_coupling", reference_id=f"{role}.shaft_datum")
        add("electrical", "motor", {
            "shaft_power_w": drv.shaft_power,
            "rpm": drv.rpm,
            "voltage_v": drv.voltage,
            "current_a": drv.current,
            "heat_w": drv.motor_heat,
        }, mass_key="motor", reference_id=None,
           status="technology_assumption",
           source_ids=["Lee et al. 2021", "Spiller, Stabile, Lentini 2013"])
        add("electrical", "inverter/controller", {
            "electric_power_w": drv.electric_power,
            "voltage_v": drv.voltage,
            "current_a": drv.current,
            "heat_w": drv.controller_heat,
        }, mass_key="inverter", reference_id=None,
           status="technology_assumption",
           source_ids=["Lee et al. 2021"])
        add("mechanical", "bearings", {
            "bearing_dn_mm_rpm": (
                line.thermal_stress.loads["bearing_dn_mm_rpm"]
                if line.thermal_stress else None
            ),
            "radial_load_n": (
                line.thermal_stress.loads["radial_load_n"]
                if line.thermal_stress else None
            ),
            "axial_thrust_n": (
                line.thermal_stress.loads["axial_thrust_n"]
                if line.thermal_stress else None
            ),
        }, mass_key="bearings", status="placeholder_screen")
        add("mechanical", "dynamic shaft seals", {
            "seal_face_speed_m_s": (
                line.thermal_stress.loads["seal_face_speed_m_s"]
                if line.thermal_stress else None
            ),
            "seal_friction_heat_w": (
                line.thermal_stress.loads["seal_friction_heat_w"]
                if line.thermal_stress else None
            ),
        }, mass_key="seals", status="placeholder_screen")
        add("pressure_boundary", "pump casing", {
            "required_outlet_pressure_pa": line.thermal_stress.loads.get("casing_pressure_pa")
            if line.thermal_stress else None,
            "wall_thickness_m": (
                line.thermal_stress.loads["casing_wall_thickness_m"]
                if line.thermal_stress else None
            ),
        }, mass_key="casing", reference_id=f"{role}.volute_scroll")
        add("interface", "inlet and outlet ports", {
            "inlet_diameter_m": imp.inlet_diameter,
            "outlet_area_m2": dif.volute_exit_area,
            "stub_length_over_diameter": 2.0,
            "wall_thickness_basis": "casing wall thickness",
        }, mass_key="ports", reference_id=f"{role}.inlet_port",
           status="screening_sized", source_ids=["NASA SP-125 eq. 8-32"])
        add("instrumentation", "pressure/temperature/speed sensors", {
            "minimum_channels": "inlet pressure, outlet pressure, motor temperature, speed pickup",
        }, status="placeholder")

    items.append(PumpHardwareBOMItem(
        role="shared",
        subsystem="electrical",
        component="battery pack / DC bus",
        quantity=1,
        status="technology_assumption",
        mass_estimate_kg=battery.mass,
        key_parameters={
            "voltage_v": battery.voltage,
            "electric_power_w": battery.electric_power,
            "current_a": battery.current,
            "heat_w": battery.heat,
            "burn_time_s": battery.burn_time,
        },
        editable_reference_id=None,
        source_ids=["Lee et al. 2021"],
    ))
    return items


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

    if spec.shared_bus and drive.voltage is not None and batt.voltage is not None:
        mismatch = abs(float(drive.voltage) - float(batt.voltage))
        gates.append(InjectorGate(
            "electric_bus_voltage_consistency",
            "pass" if mismatch <= 1.0e-6 * max(abs(float(batt.voltage)), 1.0) else "fail",
            f"shared-bus architecture requires one bus voltage; motor "
            f"{float(drive.voltage):.0f} V, battery {float(batt.voltage):.0f} V",
        ))
        if mismatch > 1.0e-6 * max(abs(float(batt.voltage)), 1.0):
            suggestions.append(
                "Use one shared --motor-voltage/--battery-voltage value, or "
                "model explicit DC/DC converters/separate packs before using "
                "different drive and pack voltages."
            )

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

        if (
            sizing.efficiency_source != "user"
            and sizing.efficiency is not None
            and sizing.efficiency >= 0.74
        ):
            gates.append(InjectorGate(
                f"pump_efficiency_screen_{role}", "warn",
                f"{role} auto pump efficiency {sizing.efficiency:.2f} is near "
                "the screening cap; replace with a measured/vendor pump curve "
                "or user-supplied efficiency before hardware decisions",
            ))

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
        width_ratio = (
            sizing.impeller.outlet_width
            / max(sizing.impeller.impeller_diameter, 1e-12)
        )
        gates.append(InjectorGate(
            f"impeller_width_ratio_{role}",
            "pass" if width_ratio <= spec.max_outlet_width_ratio else "fail",
            f"{role} b2/D2={width_ratio:.3f} vs limit "
            f"{spec.max_outlet_width_ratio:.3f}",
        ))

        channel = (
            sizing.reference_geometry.meridional_channel
            if sizing.reference_geometry is not None else None
        )
        if channel is not None:
            eye = channel.get("eye_solve") or {}
            continuity_residual = abs(
                float(eye.get("continuity_residual_m3_s", float("inf")))
            )
            continuity_tol = max(sizing.volumetric_flow, 1e-12) * 1e-9
            gates.append(InjectorGate(
                f"pump_annular_eye_continuity_{role}",
                "pass" if continuity_residual <= continuity_tol else "fail",
                f"{role} annular-eye continuity residual "
                f"{continuity_residual:.3e} m3/s vs {continuity_tol:.3e}; "
                f"effective area {channel['effective_inlet_area_m2']:.3e} m2",
            ))
            phi1 = float(eye.get("inlet_flow_coefficient", float("nan")))
            phi1_target = float(
                eye.get("target_inlet_flow_coefficient", spec.inlet_flow_coefficient)
            )
            gates.append(InjectorGate(
                f"pump_inlet_flow_coefficient_{role}",
                "pass" if math.isfinite(phi1) and abs(phi1 - phi1_target)
                <= 1e-8 * max(abs(phi1_target), 1.0) else "fail",
                f"{role} achieved phi_t1={phi1:.6g} vs target "
                f"{phi1_target:.6g} using the net annular eye area",
            ))
            b1 = float(channel.get("inlet_blockage_fraction", float("inf")))
            b2 = float(channel.get("exit_blockage_fraction", float("inf")))
            gates.append(InjectorGate(
                f"impeller_inlet_free_area_{role}",
                "pass" if b1 <= spec.max_impeller_inlet_blockage else "fail",
                f"{role} inlet blockage {b1:.3f} with "
                f"{sizing.impeller.inlet_blade_count} full blades; limit "
                f"{spec.max_impeller_inlet_blockage:.3f}",
            ))
            gates.append(InjectorGate(
                f"impeller_exit_free_area_{role}",
                "pass" if b2 <= spec.max_impeller_exit_blockage + 1e-12 else "fail",
                f"{role} exit blockage {b2:.3f} with "
                f"{sizing.impeller.blade_count} total blades; limit "
                f"{spec.max_impeller_exit_blockage:.3f}",
            ))
            if sizing.impeller.blade_root_structural_geometry_limited:
                gates.append(InjectorGate(
                    f"blade_root_geometry_closure_{role}",
                    "fail",
                    f"{role} structurally required blade-root thickness "
                    f"{sizing.impeller.blade_thickness*1e3:.3f} mm cannot fit "
                    "the exit free-area envelope at the fixed RPM or maximum "
                    f"impeller diameter {spec.max_impeller_diameter*1e3:.1f} mm",
                ))
            shaft_d = float(sizing.reference_geometry.shaft_datum["diameter_m"])
            fit = float(channel["shaft_fit_radial_clearance_m"])
            wall = float(channel["impeller_hub_wall_thickness_m"])
            hub = float(channel["eye_hub_radius_m"])
            fit_margin = hub - (0.5 * shaft_d + fit + wall)
            gates.append(InjectorGate(
                f"pump_shaft_hub_fit_{role}",
                "pass" if fit_margin >= -1e-12 else "fail",
                f"{role} solved eye hub fit margin {fit_margin*1e3:+.4f} mm "
                "after shaft, bore clearance, and root wall",
            ))
        if sizing.hydraulic_meanline is not None:
            tri = sizing.hydraulic_meanline.velocity_triangle
            gates.append(InjectorGate(
                f"impeller_design_incidence_{role}",
                "pass" if abs(tri.inlet_incidence_deg) <= 1e-9 else "fail",
                f"{role} beta1 flow={tri.inlet_relative_flow_angle_deg:.4f} deg, "
                f"metal={tri.inlet_blade_metal_angle_deg:.4f} deg, "
                f"incidence={tri.inlet_incidence_deg:+.4g} deg",
            ))
        if sizing.reference_geometry is not None:
            joint = sizing.reference_geometry.volute_scroll.get(
                "split_casing_joint", {}
            )
            gates.append(InjectorGate(
                f"split_casing_bolt_clamp_screen_{role}",
                "pass" if joint.get("bolt_screen_passed") else "fail",
                f"{role} split casing preliminary bolt stress "
                f"{float(joint.get('bolt_screen_stress_pa', float('nan')))/1e6:.1f} "
                f"MPa vs allowable "
                f"{float(joint.get('bolt_allowable_stress_pa', float('nan')))/1e6:.1f} MPa",
            ))
            gates.append(InjectorGate(
                f"split_casing_scroll_tool_access_{role}",
                "pass" if joint.get("scroll_tool_access_passed") else "fail",
                f"{role} selected scroll tool "
                f"{float(joint.get('selected_scroll_tool_diameter_m', float('nan')))*1e3:.3f} mm "
                f"vs minimum modeled scroll section "
                f"{float(joint.get('minimum_modeled_scroll_section_diameter_m', float('nan')))*1e3:.3f} mm",
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

        if sizing.architecture is not None:
            arch = sizing.architecture
            arch_status = (
                "pass" if arch.primary_type in {
                    "radial_centrifugal",
                    "radial_centrifugal_low_specific_speed",
                } else "warn"
            )
            gates.append(InjectorGate(
                f"pump_architecture_{role}", arch_status,
                f"{role} classified as {arch.primary_type} "
                f"({arch.stage_mode}, {arch.suction_assist})",
            ))

        if sizing.system_curve is not None and sizing.system_curve.points:
            worst_margin = min(p.pressure_margin for p in sizing.system_curve.points)
            design_fail = any(
                p.status == "fail" and abs(p.flow_ratio - 1.0) <= 1e-9
                for p in sizing.system_curve.points
            )
            any_fail = any(p.status == "fail" for p in sizing.system_curve.points)
            any_warn = any(p.status == "warn" for p in sizing.system_curve.points)
            gates.append(InjectorGate(
                f"system_curve_throttle_margin_{role}",
                "fail" if design_fail else (
                    "warn" if (any_fail or any_warn) else "pass"
                ),
                f"{role} fixed-speed pump curve vs throttle system curve: "
                f"worst margin {worst_margin/1e5:+.2f} bar, supported range "
                f"{sizing.system_curve.supported_throttle_range}",
            ))

        if sizing.thermal_stress is not None:
            ts = sizing.thermal_stress
            margins = ts.margins
            thermal = ts.thermal
            stress = ts.stress
            loads = ts.loads

            gates.append(InjectorGate(
                f"motor_heat_screen_{role}",
                _margin_status(margins.get("motor_heat_fraction_to_50pct"), warn=1.5),
                f"{role} drive heat fraction "
                f"{thermal['drive_heat_fraction_of_shaft_power']:.2f} of shaft power",
            ))
            gates.append(InjectorGate(
                f"propellant_heating_{role}",
                _margin_status(margins.get("propellant_delta_t"), warn=1.5),
                f"{role} estimated pump heat pickup raises propellant "
                f"{thermal['estimated_propellant_temperature_rise_k']:.2f} K",
            ))
            for key, label, value_key in (
                ("impeller_rotating_stress", "impeller rotating hoop", "impeller_rotating_hoop_stress_pa"),
                ("inducer_rotating_stress", "inducer rotating hoop", "inducer_rotating_hoop_stress_pa"),
                ("blade_root_bending", "blade-root bending", "blade_root_bending_stress_pa"),
                ("shaft_torsion", "shaft torsional shear", "shaft_torsional_shear_pa"),
                ("casing_pressure", "casing pressure hoop", "casing_hoop_stress_pa"),
            ):
                margin = margins.get(key)
                gates.append(InjectorGate(
                    f"{key}_{role}",
                    _margin_status(margin),
                    f"{role} {label} screen margin "
                    f"{'n/a' if margin is None else f'{margin:.2f}'} "
                    f"(stress {stress[value_key]/1e6:.1f} MPa)",
                ))
            for key, label, value_key, unit in (
                ("bearing_dn", "bearing DN", "bearing_dn_mm_rpm", "mm-rpm"),
                ("seal_face_speed", "seal face speed", "seal_face_speed_m_s", "m/s"),
            ):
                margin = margins.get(key)
                gates.append(InjectorGate(
                    f"{key}_{role}",
                    _margin_status(margin),
                    f"{role} {label} margin "
                    f"{'n/a' if margin is None else f'{margin:.2f}'} "
                    f"({loads[value_key]:.3g} {unit})",
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


def _solve_coupled_line_geometry(
    role: str,
    ln,
    head: float,
    hydraulic_power: float,
    initial_rpm: float,
    initial_efficiency: float,
    efficiency_source: str,
    spec: PumpSizingSpec,
) -> tuple[
    CentrifugalPumpGeometry,
    InducerGeometry,
    PumpHydraulicMeanline,
    dict,
    float,
    float,
    float,
]:
    """Close blade free area around the shaft/eye/meanline fixed point."""
    rpm_trial = float(initial_rpm)
    user_efficiency = efficiency_source == "user"
    blade_root_structural_floor = 0.0
    for free_area_iteration in range(1, 31):
        impeller = _impeller_geometry(
            role,
            ln.volumetric_flow,
            head,
            rpm_trial,
            spec,
            blade_root_structural_floor=blade_root_structural_floor,
        )
        shaft_guess = hydraulic_power / max(initial_efficiency, 1e-9)
        shaft_d = _shaft_diameter(
            impeller, spec, shaft_power=shaft_guess
        )
        channel = None
        eye_solve = None
        for coupled_iteration in range(1, 26):
            prior = (
                impeller.inlet_diameter,
                shaft_d,
                shaft_guess,
            )
            eye_solve = _solve_annular_eye_and_shaft(
                ln.volumetric_flow,
                impeller,
                spec,
                shaft_power=shaft_guess,
            )
            shaft_d = float(eye_solve["shaft_diameter_m"])
            inducer = _inducer_geometry(
                role,
                ln.volumetric_flow,
                ln,
                impeller,
                spec,
                shaft_diameter=shaft_d,
            )
            length = max(
                1.8 * impeller.impeller_diameter,
                inducer.diameter + impeller.impeller_diameter,
            )
            channel = _meridional_channel(
                role,
                ln.volumetric_flow,
                impeller,
                inducer,
                0.08 * length,
                shaft_diameter=shaft_d,
                spec=spec,
                eye_solve=eye_solve,
            )
            meanline = _hydraulic_meanline(
                role,
                ln.volumetric_flow,
                head,
                ln,
                impeller,
                spec,
                inlet_area=channel["effective_inlet_area_m2"],
            )
            eta_next = (
                initial_efficiency
                if user_efficiency
                else meanline.hydraulic_efficiency
            )
            shaft_next = hydraulic_power / max(eta_next, 1e-9)
            next_shaft_d = _shaft_diameter(
                impeller, spec, shaft_power=shaft_next
            )
            change = max(
                abs(impeller.inlet_diameter - prior[0])
                / max(impeller.inlet_diameter, 1e-12),
                abs(next_shaft_d - prior[1])
                / max(next_shaft_d, 1e-12),
                abs(shaft_next - prior[2])
                / max(shaft_next, 1.0),
            )
            shaft_guess = shaft_next
            shaft_d = next_shaft_d
            if change <= 1.0e-10:
                break
        else:
            raise RuntimeError(
                f"{role} coupled shaft/eye/meanline solve did not converge"
            )
        assert channel is not None and eye_solve is not None
        # Close the existing blade-root cantilever screen back onto the actual
        # blade thickness before accepting the hydraulic geometry.  With
        # Z blades, F_blade=rho*Q*|Vtheta2|/Z and the rectangular root section
        # used by the stress ledger gives sigma=6*F_blade/t^2.  The allowable
        # already includes structural_fos.  If the required root grows, the
        # next outer iteration re-closes exit free area and RPM/D2 around it.
        rotor_allowable = (
            spec.rotor_yield_strength / max(spec.structural_fos, 1.0e-12)
        )
        blade_force_total = (
            max(float(ln.density), 1.0e-12)
            * ln.volumetric_flow
            * abs(meanline.velocity_triangle.outlet_whirl_velocity)
        )
        blade_force_each = blade_force_total / max(impeller.blade_count, 1)
        required_root_thickness = math.sqrt(
            6.0 * blade_force_each / max(rotor_allowable, 1.0e-12)
        )
        impeller.blade_root_structural_minimum_thickness = (
            required_root_thickness
        )
        current_blade_thickness = float(impeller.blade_thickness or 0.0)
        if required_root_thickness > current_blade_thickness * (1.0 + 1.0e-10):
            blade_root_structural_floor = max(
                blade_root_structural_floor,
                required_root_thickness * (1.0 + 1.0e-9),
            )
            continue
        inlet_blockage = float(eye_solve["inlet_blockage_fraction"])
        exit_blockage = float(impeller.exit_blockage_fraction or 0.0)
        if (
            inlet_blockage <= spec.max_impeller_inlet_blockage + 1e-12
            and exit_blockage <= spec.max_impeller_exit_blockage + 1e-12
        ):
            channel["coupled_mechanical_hydraulic_iterations"] = (
                coupled_iteration
            )
            channel["coupled_mechanical_hydraulic_converged"] = True
            channel["blade_free_area_iterations"] = free_area_iteration
            channel["blade_free_area_converged"] = True
            channel["shaft_diameter_m"] = shaft_d
            impeller.inlet_blade_angle_deg = (
                meanline.velocity_triangle.inlet_blade_metal_angle_deg
            )
            eta = (
                initial_efficiency
                if user_efficiency else meanline.hydraulic_efficiency
            )
            shaft_power = hydraulic_power / max(eta, 1e-9)
            return (
                impeller,
                inducer,
                meanline,
                channel,
                shaft_d,
                eta,
                shaft_power,
            )
        if (
            spec.drive.rpm is not None
            or impeller.blade_root_structural_geometry_limited
        ):
            channel["coupled_mechanical_hydraulic_iterations"] = (
                coupled_iteration
            )
            channel["coupled_mechanical_hydraulic_converged"] = True
            channel["blade_free_area_iterations"] = free_area_iteration
            channel["blade_free_area_converged"] = False
            channel["shaft_diameter_m"] = shaft_d
            impeller.inlet_blade_angle_deg = (
                meanline.velocity_triangle.inlet_blade_metal_angle_deg
            )
            eta = (
                initial_efficiency
                if user_efficiency else meanline.hydraulic_efficiency
            )
            shaft_power = hydraulic_power / max(eta, 1e-9)
            return (
                impeller, inducer, meanline, channel, shaft_d, eta, shaft_power
            )
        # Lower speed grows D2/D1 at unchanged U2.  A damped ratio update is
        # monotone for this similarity solve and avoids overshooting width
        # and maximum-diameter constraints.
        ratio = (
            spec.max_impeller_inlet_blockage
            / max(inlet_blockage, 1e-12)
        )
        rpm_trial = impeller.rpm * _clamp(0.90 * ratio, 0.35, 0.90)
    raise RuntimeError(
        f"{role} impeller blade free-area solve did not converge"
    )


def size_electric_pumps(
    ledger: FeedSystemLedger,
    spec: PumpSizingSpec | None = None,
) -> ElectricPumpSizingResult:
    """Size electric pump feed hardware from an injector feed-system ledger."""
    spec = spec or PumpSizingSpec()
    lines: dict[str, PumpLineSizing] = {}
    pending_drive: dict[str, tuple[float, float]] = {}
    line_electric_power: dict[str, float] = {}
    eta_m = max(1e-6, min(1.0, spec.drive.motor_efficiency))
    eta_i = max(1e-6, min(1.0, spec.drive.inverter_efficiency))
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
        architecture = None
        reference_geometry = None
        system_curve = None
        if _finite(ln.required_pressure_rise):
            rise = max(0.0, float(ln.required_pressure_rise))
            hydraulic = ln.volumetric_flow * rise
            head = ln.required_pump_head if ln.required_pump_head is not None else 0.0
            rpm, rpm_source = _select_rpm(ln.volumetric_flow, head, spec)
            (
                impeller,
                inducer,
                meanline,
                channel,
                shaft_d,
                eta,
                shaft,
            ) = _solve_coupled_line_geometry(
                role,
                ln,
                head,
                hydraulic,
                rpm,
                eta,
                eta_source,
                spec,
            )
            if eta_source != "user":
                eta_source = meanline.efficiency_source
            rpm = impeller.rpm
            rpm_source = (
                f"{rpm_source}; coupled shaft/annular-eye/blade-free-area "
                f"solve ({channel['blade_free_area_iterations']} outer "
                "iterations)"
            )
            pending_drive[role] = (shaft, rpm)
            line_electric_power[role] = shaft / max(eta_m * eta_i, 1e-12)
            diffuser = _diffuser_volute_geometry(role, ln.volumetric_flow, impeller, spec)
            curve = _pump_performance_curve(
                role, ln.volumetric_flow, head, ln.density, rpm, eta
            )
            architecture = _architecture_classification(
                role, ln, impeller, inducer, spec
            )
            reference_geometry = _reference_geometry(
                role,
                ln,
                impeller,
                inducer,
                diffuser,
                spec,
                meanline,
                channel=channel,
                shaft_diameter=shaft_d,
            )
            system_curve = _system_curve_coupling(role, ln, curve)
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
            architecture=architecture,
            reference_geometry=reference_geometry,
            system_curve=system_curve,
        )

    if spec.shared_bus:
        selected_bus_voltage, selected_bus_source = _select_shared_bus_voltage(
            line_electric_power, spec
        )
    else:
        selected_bus_voltage = 0.0
        selected_bus_source = "per_stream_drive_bus"
    total_electric = 0.0
    for role, (shaft, rpm) in pending_drive.items():
        if spec.shared_bus:
            drive = _drive_sizing(
                role, shaft, rpm, spec,
                bus_voltage=selected_bus_voltage,
                voltage_source=selected_bus_source,
            )
        else:
            drive = _drive_sizing(role, shaft, rpm, spec)
            selected_bus_voltage = max(selected_bus_voltage, drive.voltage)
        lines[role].drive = drive
        total_electric += drive.electric_power
    if selected_bus_voltage <= 0.0:
        selected_bus_voltage = 48.0
        selected_bus_source = "auto_default_no_pump_duty"
    battery = _battery_sizing(total_electric, spec, selected_bus_voltage)
    for role, line in lines.items():
        line.thermal_stress = _thermal_stress_ledger(
            role, ledger.lines[role], line, spec
        )
    feasibility = _feasibility(ledger, lines, battery, spec)
    hardware_bom = _hardware_bom(lines, battery, spec)
    assumptions = {
        "burn_time_s": spec.burn_time,
        "pump_rpm": spec.drive.rpm if spec.drive.rpm is not None else "auto",
        "electric_bus_architecture": (
            "shared_pack_bus" if spec.shared_bus else "per_stream_drive_bus"
        ),
        "selected_bus_voltage_v": selected_bus_voltage,
        "selected_bus_voltage_source": selected_bus_source,
        "motor_voltage_v": spec.drive.voltage if spec.drive.voltage is not None else "auto",
        "battery_voltage_v": selected_bus_voltage if spec.shared_bus else (
            spec.battery.voltage or spec.drive.voltage or selected_bus_voltage
        ),
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
        "blade_root_structural_closure": (
            "t_root>=sqrt(6*F_blade/(rotor_yield/structural_fos)); "
            "solved thickness is re-closed through exit blockage and RPM/D2"
        ),
        "material_tip_speed_limit_m_s": spec.material_tip_speed_limit,
        "rotor_material_density_kg_m3": spec.rotor_material_density,
        "rotor_yield_strength_pa": spec.rotor_yield_strength,
        "casing_yield_strength_pa": spec.casing_yield_strength,
        "structural_fos": spec.structural_fos,
        "bearing_dn_limit_mm_rpm": spec.bearing_dn_limit,
        "seal_face_speed_limit_m_s": spec.seal_face_speed_limit,
        "max_propellant_temperature_rise_k": spec.max_propellant_temperature_rise,
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
            "pump curve and feed-system curve are compared over throttle flow ratios",
            "loss heat, rotor speed, pressure loads, shaft torque, bearing DN, and seal speed create first-pass thermal/stress gates",
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
        hardware_bom=hardware_bom,
        notes=notes,
    )
