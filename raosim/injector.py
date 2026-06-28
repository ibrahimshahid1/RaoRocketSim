"""injector.py - Parameterized pintle-injector hydraulic sizing.

The injector is sized from the operating point the nozzle + chamber solver
already produces, never from an independent injector thrust input:

    F, Pc, Pa, eps, O/F  ->  Cf, At, mdot  ->  mdot_f, mdot_o
                          ->  A_f, A_o, dp_f, dp_o  ->  pintle geometry + spray

Core incompressible orifice hydraulics (Sutton & Biblarz, *Rocket Propulsion
Elements*; NASA SP-8089 *Liquid Rocket Engine Injectors*):

    mdot = Cd * A * sqrt(2 * rho * dp)            (per stream)
    v    = mdot / (rho * A)                        (mean passage velocity)
    TMR  = (mdot_r * v_r) / (mdot_a * v_a)         (radial / axial momentum)
    Re   = rho * v * D_h / mu
    We   = rho * v**2 * D_h / sigma
    Oh   = mu / sqrt(rho * sigma * D_h)            ( = sqrt(We) / Re )

Pintle passage geometry:
    axial annulus      A_a = (pi/4) (Do^2 - Di^2)  ~ pi * D_p * h
    radial rect slots  A_r = N_slots * w * h_slot

This first implementation supports a FIXED, LIQUID/LIQUID, automatically-sized
pintle.  Gaseous / near-critical injection needs a separate compressible /
real-fluid branch and is explicitly rejected here.  Movable-sleeve throttling
is a later implementation.

NASA SP-8089 identifies momentum balance, annulus/slot geometry, and the
deflector angle as the governing pintle variables, while warning that spray
distributions generally require cold-flow testing — so every spray/atomization
number below is a clearly-labeled screening surrogate, not a validated result.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from raosim.coolants import canonical_coolant_name


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
@dataclass
class PropellantFeedSpec:
    """Inlet (feed) state of one propellant entering the injector."""

    role: str = "fuel"                       # "fuel" | "oxidizer"
    name: str | None = None                  # e.g. "LOX", "RP-1", "methane"
    inlet_temperature: float | None = None   # K  (None -> per-fluid default)
    inlet_pressure: float | None = None      # Pa (None -> manifold requirement)
    phase: str = "auto"                       # "auto" | "liquid" | "gas"
    # Optional explicit property overrides (bypass the property backend).
    density: float | None = None             # kg/m^3
    viscosity: float | None = None           # Pa.s
    surface_tension: float | None = None     # N/m
    vapor_pressure: float | None = None      # Pa
    property_source: str | None = None       # filled by the resolver


@dataclass
class FeedLineSpec:
    """Per-propellant feed-system inputs for the pump/tank pressure closure.

    The pintle sets only the injector metering drop; the propellant still has to
    be delivered against chamber pressure plus every downstream loss.  This
    captures the rest of the feed budget so the tool can answer whether the
    chosen pump/tank can actually supply the geometry it sized (Huzel & Huang,
    *Design of Liquid Propellant Rocket Engines*, NASA SP-125, Ch. 1-2; Sutton &
    Biblarz, *Rocket Propulsion Elements*, feed-system chapter; inducer/NPSH per
    NASA SP-8052).

    All pressures are absolute Pa.  Fields left ``None`` are unknown: the ledger
    then reports the *requirement* as an info gate instead of judging a pump it
    was never told about.
    """

    # Available pump-discharge (pump-fed) or tank-ullage (pressure-fed) pressure
    # at the propellant outlet, upstream of the losses below.  Pa.
    supply_pressure: float | None = None
    # Delivered mass-flow capacity of the pump/feed line.  kg/s.
    flow_capacity: float | None = None
    # Lumped line + valve + filter loss between supply and injector manifold,
    # as an absolute pressure and/or a fraction of Pc (the two are summed).
    line_loss: float = 0.0                    # Pa
    line_loss_fraction: float = 0.0           # x Pc
    # Injector manifold (header + port distribution) loss ALLOWANCE charged to
    # the pump budget, absolute and/or fraction of Pc (summed).  Defaults to 0:
    # the maldistribution-network estimate is reported separately as
    # ``manifold_screen_loss`` (informational, requires 3-D validation) and is
    # NOT charged automatically, so an unvalidated screen value cannot silently
    # dominate the pump requirement.  Set this from that screen once trusted.
    manifold_loss: float = 0.0                # Pa
    manifold_loss_fraction: float = 0.0       # x Pc
    # Control / transient / mixture-ratio-trim margin held above the steady
    # requirement (absolute and/or fraction of Pc, summed).
    control_margin: float = 0.0               # Pa
    control_margin_fraction: float = 0.0      # x Pc
    # Suction side, for the NPSH screen: tank/inlet stagnation pressure and the
    # pump's required net positive suction pressure (both Pa).
    tank_pressure: float | None = None
    npsh_required: float | None = None
    # Assumed pump efficiency for the ideal shaft-power estimate (0, 1].
    pump_efficiency: float = 0.7


@dataclass
class FeedSystemSpec:
    """Both propellant feed lines plus the feed-architecture label."""

    architecture: str = "pump_fed"            # "pump_fed" | "pressure_fed"
    fuel: FeedLineSpec = field(default_factory=lambda: FeedLineSpec())
    oxidizer: FeedLineSpec = field(default_factory=lambda: FeedLineSpec())


@dataclass
class PintleGeometrySpec:
    """Pintle geometric anchors + manufacturing-driven slot division."""

    pintle_diameter: float | None = None     # m  (annulus + slot anchor)
    slot_count: int = 24                     # number of radial openings
    slot_aspect_ratio: float = 1.0           # h_slot / w
    deflector_angle: float = 0.0             # deg, radial-stream deflection
    impingement_distance: float | None = None  # m, openings -> interaction
    radial_stream: str = "fuel"              # which stream is slotted (radial)
    slot_length_over_dh: float = 2.0         # auto slot depth = (L/D)*D_h
    # Fixed-geometry overrides (only honoured under sizing="fixed").
    annulus_gap: float | None = None         # m
    slot_width: float | None = None          # m
    slot_height: float | None = None         # m
    slot_depth: float | None = None          # m
    tip_radius: float | None = None          # m
    body_length: float | None = None         # m
    face_thickness: float | None = None      # m
    face_od: float | None = None             # m


@dataclass
class InjectorManufacturingSpec:
    """Manufacturing floors for gaps, slots and ligaments."""

    min_feature: float = 3.0e-4              # m  (0.3 mm default floor)
    web_min: float | None = None            # m  (ligament; default min_feature)
    edge_distance_min: float | None = None  # m  (default min_feature)
    concentricity_tolerance: float = 5.0e-5  # m  (annulus eccentricity, 50 um)


@dataclass
class InjectorSpec:
    """Top-level injector request attached to a DesignInput."""

    type: str = "none"                       # "none" | "pintle"
    sizing: str = "auto"                     # "auto" | "fixed"
    fuel_dp_fraction: float = 0.2            # dp_f / Pc
    oxidizer_dp_fraction: float = 0.2        # dp_o / Pc
    fuel_cd: float = 0.7                     # fuel metering discharge coeff
    oxidizer_cd: float = 0.7                 # oxidizer metering discharge coeff
    faceplate_material: str | None = None
    pintle_material: str | None = None
    target_momentum_ratio: float | None = None
    # Discrete feed-port count for each propellant manifold (annular header
    # ring → annulus/slots). Drives the maldistribution network.
    fuel_manifold_ports: int = 4
    oxidizer_manifold_ports: int = 4
    # d^2-law droplet burning-rate constant [m^2/s] used by the vaporization /
    # combustion-development screen (hydrocarbon class ~1e-6; screening-grade).
    evaporation_constant: float = 1.0e-6
    # Failed injector gates block integrated design/CAD workflows unless this
    # explicit preliminary-analysis override is selected.
    allow_infeasible: bool = False
    fuel: PropellantFeedSpec = field(
        default_factory=lambda: PropellantFeedSpec(role="fuel")
    )
    oxidizer: PropellantFeedSpec = field(
        default_factory=lambda: PropellantFeedSpec(role="oxidizer")
    )
    geometry: PintleGeometrySpec = field(default_factory=PintleGeometrySpec)
    manufacturing: InjectorManufacturingSpec = field(
        default_factory=InjectorManufacturingSpec
    )
    # Pump/tank feed-system inputs for the feed-pressure closure (§ feed ledger).
    feed_system: FeedSystemSpec = field(default_factory=FeedSystemSpec)
    # Injector CAD/reference-geometry output: "none" | "reference" | "parts";
    # format "step" (portable B-rep) | "stl" (mesh) | "dxf" (2-D profile).
    cad: str = "none"
    cad_format: str = "step"


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass
class FeedState:
    """Resolved liquid feed properties used for hydraulic sizing."""

    role: str
    name: str
    temperature: float          # K
    pressure: float             # Pa  (property-evaluation pressure)
    density: float              # kg/m^3
    viscosity: float            # Pa.s
    surface_tension: float      # N/m
    vapor_pressure: float       # Pa
    phase: str                  # "liquid" | "gas" | "supercritical"
    critical_temperature: float | None
    critical_pressure: float | None
    source: str
    liquid_ok: bool
    reason: str = ""
    # Compressible (gas / supercritical) branch.  ``gas_ok`` marks a state the
    # compressible orifice equations can size; gamma + specific gas constant
    # are required to do so.
    gas_ok: bool = False
    gamma: float | None = None          # cp/cv of the injected gas
    gas_constant: float | None = None   # R_specific = R_u/Mw  [J/(kg·K)]
    # Thermal properties used by the face/tip cooling screen.
    cp: float | None = None             # specific heat  [J/(kg·K)]
    conductivity: float | None = None   # thermal conductivity  [W/(m·K)]


@dataclass
class StreamResult:
    role: str
    geometry: str               # "annulus" | "slots"
    mdot: float                 # kg/s
    dp: float                   # Pa
    cd: float
    area: float                 # m^2 (geometric flow area)
    velocity: float             # m/s
    hydraulic_diameter: float   # m
    reynolds: float
    weber: float
    ohnesorge: float
    # geometry-specific (filled where relevant)
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamAtomization:
    """Per-stream primary-atomization + vaporization screening estimate."""

    role: str
    sauter_mean_diameter: float     # m, d_32 (aerodynamic / Hinze limit)
    aerodynamic_weber: float        # We_g = rho_g v^2 d_jet / sigma
    breakup_length: float           # m, primary breakup length L_b≈15·d_jet (Reitz & Bracco)
    vaporization_length: float      # m, d^2-law length to ~99% vaporized
    combustion_length: float        # m, breakup + vaporization
    vaporized_fraction: float       # [-], fraction vaporized in the chamber
    regime: str                     # atomization-regime validity flag


@dataclass
class SprayAtomization:
    """Spray atomization / vaporization screen for the whole injector.

    All numbers are clearly-labeled order-of-magnitude SURROGATES (SP-8089:
    pintle spray distributions require cold-flow testing); they exist to flag
    when combustion development cannot fit the available chamber length and to
    drive the L*/injector-quality coupling, not to predict performance.
    """

    chamber_gas_density: float          # kg/m^3, Pc/(R_gas Tc)
    evaporation_constant: float         # m^2/s, d^2-law burning-rate constant
    streams: dict[str, StreamAtomization]
    limiting_role: str                  # stream with the worst (longest) dev.
    combustion_length: float            # m, limiting stream
    available_chamber_length: float     # m
    development_margin: float           # available / required (>=1 good)
    predicted_cstar_efficiency: float   # mass-weighted vaporized fraction
    model: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "chamber_gas_density_kg_m3": self.chamber_gas_density,
            "evaporation_constant_m2_s": self.evaporation_constant,
            "limiting_role": self.limiting_role,
            "combustion_length_m": self.combustion_length,
            "available_chamber_length_m": self.available_chamber_length,
            "development_margin": self.development_margin,
            "predicted_cstar_efficiency": self.predicted_cstar_efficiency,
            "model": self.model,
            "streams": {
                role: {
                    "sauter_mean_diameter_m": s.sauter_mean_diameter,
                    "aerodynamic_weber": s.aerodynamic_weber,
                    "breakup_length_m": s.breakup_length,
                    "vaporization_length_m": s.vaporization_length,
                    "combustion_length_m": s.combustion_length,
                    "vaporized_fraction": s.vaporized_fraction,
                    "regime": s.regime,
                }
                for role, s in self.streams.items()
            },
            "notes": self.notes,
        }


@dataclass
class ManifoldResult:
    """Distribution of one propellant manifold into its injection elements."""

    role: str
    feeds: str                       # "annulus" | "slots"
    element_count: int               # slots, or annulus discretization segments
    port_count: int                  # discrete manifold feed ports
    maldistribution_fraction: float  # (max-min)/mean element flow
    min_flow_ratio: float
    max_flow_ratio: float
    manifold_pressure_drop: float    # Pa, header + port losses
    port_diameter: float             # m
    plenum_area: float               # m^2
    status: str


@dataclass
class ManifoldDistribution:
    streams: dict[str, ManifoldResult]
    worst_maldistribution: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "worst_maldistribution": self.worst_maldistribution,
            "streams": {
                role: {
                    "feeds": m.feeds, "element_count": m.element_count,
                    "port_count": m.port_count,
                    "maldistribution_fraction": m.maldistribution_fraction,
                    "min_flow_ratio": m.min_flow_ratio,
                    "max_flow_ratio": m.max_flow_ratio,
                    "manifold_pressure_drop_pa": m.manifold_pressure_drop,
                    "port_diameter_m": m.port_diameter,
                    "plenum_area_m2": m.plenum_area, "status": m.status,
                }
                for role, m in self.streams.items()
            },
            "notes": self.notes,
        }


@dataclass
class FaceTipThermal:
    """Screening face / pintle-tip heat balance."""

    recovery_temperature: float        # K, recirculation recovery temperature
    tip_gas_coefficient: float         # W/(m^2 K)
    tip_coolant_coefficient: float     # W/(m^2 K)
    tip_heat_flux: float               # W/m^2
    tip_wall_temperature: float        # K, gas-side
    tip_margin: float                  # T_limit / T_wg
    face_heat_flux: float              # W/m^2
    face_wall_temperature: float       # K, gas-side
    face_margin: float
    limiting: str                      # "tip" | "face"
    wall_temperature_limit: float      # K
    governing_margin: float
    model: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "recovery_temperature_K": self.recovery_temperature,
            "wall_temperature_limit_K": self.wall_temperature_limit,
            "tip": {
                "gas_coefficient_W_m2K": self.tip_gas_coefficient,
                "coolant_coefficient_W_m2K": self.tip_coolant_coefficient,
                "heat_flux_W_m2": self.tip_heat_flux,
                "wall_temperature_K": self.tip_wall_temperature,
                "margin": self.tip_margin,
            },
            "face": {
                "heat_flux_W_m2": self.face_heat_flux,
                "wall_temperature_K": self.face_wall_temperature,
                "margin": self.face_margin,
            },
            "limiting": self.limiting,
            "governing_margin": self.governing_margin,
            "model": self.model,
            "notes": self.notes,
        }


@dataclass
class StabilityScreen:
    """Feed-system + chamber-acoustic + n-τ combustion stability screen."""

    sound_speed: float                  # m/s, chamber gas
    f_L1: float                         # Hz, first longitudinal
    f_L2: float
    f_T1: float                         # Hz, first tangential
    f_R1: float                         # Hz, first radial
    injector_decoupling_fraction: float # min(χ_f, χ_o)
    chug_status: str
    combustion_time_lag: float          # s
    reduced_frequency_L1: float         # τ·f_L1
    sensitive_band: bool                # in the n-τ instability band
    model: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "sound_speed_m_s": self.sound_speed,
            "f_L1_Hz": self.f_L1, "f_L2_Hz": self.f_L2,
            "f_T1_Hz": self.f_T1, "f_R1_Hz": self.f_R1,
            "injector_decoupling_fraction": self.injector_decoupling_fraction,
            "chug_status": self.chug_status,
            "combustion_time_lag_s": self.combustion_time_lag,
            "reduced_frequency_L1": self.reduced_frequency_L1,
            "sensitive_band": self.sensitive_band,
            "model": self.model, "notes": self.notes,
        }


@dataclass
class InjectorGate:
    name: str
    status: str                 # "pass" | "warn" | "fail" | "info"
    detail: str

    @property
    def ok(self) -> bool:
        return self.status in ("pass", "info")


@dataclass
class FeedLineLedger:
    """Resolved feed-pressure budget and pump duty for one propellant."""

    role: str
    chamber_pressure: float                   # Pa
    injector_dp: float                        # Pa  (metering drop across orifice)
    manifold_loss: float                      # Pa  (charged header/port allowance)
    manifold_screen_loss: float               # Pa  (maldistribution-network estimate, info only)
    regen_loss: float                         # Pa  (jacket dP if stream cools first)
    line_valve_loss: float                    # Pa
    control_margin: float                     # Pa
    required_outlet_pressure: float           # Pa  (sum of the above)
    available_outlet_pressure: float | None   # Pa
    pressure_margin: float | None             # Pa  (available - required)
    density: float                            # kg/m^3
    volumetric_flow: float                    # m^3/s  (mdot / rho)
    required_pressure_rise: float | None      # Pa  max(required_outlet - tank, 0)
    required_pump_head: float | None          # m
    ideal_pump_power: float | None            # W   (Q * rise / eta_pump)
    flow_capacity: float | None               # kg/s
    capacity_margin: float | None             # kg/s
    npsh_available: float | None              # Pa
    npsh_required: float | None               # Pa
    npsh_margin: float | None                 # Pa
    status: str                               # pass | warn | fail | info

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "chamber_pressure_pa": self.chamber_pressure,
            "injector_dp_pa": self.injector_dp,
            "manifold_loss_pa": self.manifold_loss,
            "manifold_screen_loss_pa": self.manifold_screen_loss,
            "regen_loss_pa": self.regen_loss,
            "line_valve_loss_pa": self.line_valve_loss,
            "control_margin_pa": self.control_margin,
            "required_outlet_pressure_pa": self.required_outlet_pressure,
            "available_outlet_pressure_pa": self.available_outlet_pressure,
            "pressure_margin_pa": self.pressure_margin,
            "density_kg_m3": self.density,
            "volumetric_flow_m3_s": self.volumetric_flow,
            "required_pressure_rise_pa": self.required_pressure_rise,
            "required_pump_head_m": self.required_pump_head,
            "ideal_pump_power_w": self.ideal_pump_power,
            "flow_capacity_kg_s": self.flow_capacity,
            "capacity_margin_kg_s": self.capacity_margin,
            "npsh_available_pa": self.npsh_available,
            "npsh_required_pa": self.npsh_required,
            "npsh_margin_pa": self.npsh_margin,
            "status": self.status,
        }


@dataclass
class FeedSystemLedger:
    """Whole-engine feed-pressure closure across both propellants."""

    architecture: str
    lines: dict[str, FeedLineLedger]
    governing_required_pressure: float        # Pa  (max across streams)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "architecture": self.architecture,
            "governing_required_pressure_pa": self.governing_required_pressure,
            "lines": {k: v.to_dict() for k, v in self.lines.items()},
            "notes": self.notes,
        }


@dataclass
class InjectorDesignResult:
    feasible: bool
    sizing: str
    radial_stream: str
    pintle_diameter: float
    slot_count: int
    chamber_radius: float
    chamber_length: float
    streams: dict[str, StreamResult]      # "fuel" / "oxidizer"
    annulus: StreamResult
    slots: StreamResult
    total_momentum_ratio: float
    spray_half_angle_deg: float
    spray_wall_axial_distance: float      # m, tip -> wall interception
    slot_to_annulus_width_ratio: float
    blockage_factor: float
    minimum_web: float                    # m, ligament between slots
    gates: list[InjectorGate]
    feed: dict[str, FeedState]
    atomization: SprayAtomization | None = None
    manifold: ManifoldDistribution | None = None
    thermal: FaceTipThermal | None = None
    stability: StabilityScreen | None = None
    feed_system: FeedSystemLedger | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        def _stream(s: StreamResult) -> dict:
            return {
                "role": s.role, "geometry": s.geometry, "mdot_kg_s": s.mdot,
                "dp_pa": s.dp, "cd": s.cd, "area_m2": s.area,
                "velocity_m_s": s.velocity, "hydraulic_diameter_m":
                s.hydraulic_diameter, "reynolds": s.reynolds,
                "weber": s.weber, "ohnesorge": s.ohnesorge, "detail": s.detail,
            }

        def _feed(f: FeedState) -> dict:
            return {
                "role": f.role, "name": f.name, "temperature_K": f.temperature,
                "pressure_pa": f.pressure, "density_kg_m3": f.density,
                "viscosity_pa_s": f.viscosity, "surface_tension_n_m":
                f.surface_tension, "vapor_pressure_pa": f.vapor_pressure,
                "phase": f.phase, "liquid_ok": f.liquid_ok,
                "source": f.source, "reason": f.reason,
            }

        return {
            "feasible": self.feasible,
            "sizing": self.sizing,
            "radial_stream": self.radial_stream,
            "pintle_diameter_m": self.pintle_diameter,
            "chamber_radius_m": self.chamber_radius,
            "chamber_length_m": self.chamber_length,
            "annulus": _stream(self.annulus),
            "slots": _stream(self.slots),
            "total_momentum_ratio": self.total_momentum_ratio,
            "spray_half_angle_deg": self.spray_half_angle_deg,
            "spray_wall_axial_distance_m": self.spray_wall_axial_distance,
            "slot_to_annulus_width_ratio": self.slot_to_annulus_width_ratio,
            "blockage_factor": self.blockage_factor,
            "minimum_web_m": self.minimum_web,
            "atomization": (
                self.atomization.to_dict() if self.atomization else None
            ),
            "manifold": (
                self.manifold.to_dict() if self.manifold else None
            ),
            "thermal": (
                self.thermal.to_dict() if self.thermal else None
            ),
            "stability": (
                self.stability.to_dict() if self.stability else None
            ),
            "feed_system": (
                self.feed_system.to_dict() if self.feed_system else None
            ),
            "feed": {k: _feed(v) for k, v in self.feed.items()},
            "gates": [
                {"name": g.name, "status": g.status, "detail": g.detail}
                for g in self.gates
            ],
            "notes": self.notes,
        }


class InjectorUnsupportedState(Exception):
    """Raised when a feed state is outside the liquid/liquid MVP envelope."""


class InjectorSpecError(ValueError):
    """Raised for an invalid injector spec, before any property resolution."""


# Mass-flow closure tolerances on delivered vs required flow.  Independent of
# sizing mode: auto sizing closes to machine precision (pass); a supplied
# fixed geometry that misses the cycle flow by more than FAIL_TOL is a real
# infeasibility, not a warning.
_CLOSURE_PASS_TOL = 1.0e-3   # < 0.1% delivered-vs-required flow error
_CLOSURE_FAIL_TOL = 0.05     # >= 5% flow error fails; between -> warn
_REGEN_FLOW_PASS_TOL = 0.01  # direct jacket->injector handoff within 1%
_REGEN_FLOW_FAIL_TOL = 0.05  # >=5% needs a bypass/mixing model


def _validate_injector_spec(
    spec,
    mdot_fuel,
    mdot_oxidizer,
    Pc,
    mixture_ratio,
    chamber_radius,
    chamber_length,
    gamma,
    Tc,
    R_gas,
):
    """Front gate: reject a malformed spec before resolving feed properties.

    Catches the cases that would otherwise raise ZeroDivisionError /
    uncaught ValueError downstream (zero Cd, zero slots, zero dp), and the
    silent fixed-mode-without-dimensions fall-through to auto sizing.
    """
    errs: list[str] = []
    if spec.type != "pintle":
        errs.append("injector type must be 'pintle'")
    if spec.sizing not in ("auto", "fixed"):
        errs.append("injector sizing must be 'auto' or 'fixed'")
    if not (0.0 < spec.fuel_cd <= 1.0):
        errs.append(f"fuel Cd must be in (0, 1], got {spec.fuel_cd}")
    if not (0.0 < spec.oxidizer_cd <= 1.0):
        errs.append(f"oxidizer Cd must be in (0, 1], got {spec.oxidizer_cd}")
    if spec.fuel_dp_fraction <= 0.0:
        errs.append(f"fuel dp-fraction must be > 0, got {spec.fuel_dp_fraction}")
    if spec.oxidizer_dp_fraction <= 0.0:
        errs.append(
            f"oxidizer dp-fraction must be > 0, got {spec.oxidizer_dp_fraction}")
    geo = spec.geometry
    if int(geo.slot_count) < 1:
        errs.append(f"slot_count must be >= 1, got {geo.slot_count}")
    if geo.slot_aspect_ratio <= 0.0:
        errs.append(f"slot_aspect_ratio must be > 0, got {geo.slot_aspect_ratio}")
    if geo.pintle_diameter is not None and geo.pintle_diameter <= 0.0:
        errs.append(f"pintle_diameter must be > 0, got {geo.pintle_diameter}")
    if geo.radial_stream not in ("fuel", "oxidizer"):
        errs.append("radial_stream must be 'fuel' or 'oxidizer'")
    if not (0.0 <= geo.deflector_angle <= 90.0):
        errs.append(
            f"deflector_angle must be in [0, 90] deg, got {geo.deflector_angle}"
        )
    if (
        geo.impingement_distance is not None
        and geo.impingement_distance < 0.0
    ):
        errs.append("impingement_distance must be >= 0")
    for name, value in (
        ("annulus_gap", geo.annulus_gap),
        ("slot_width", geo.slot_width),
        ("slot_height", geo.slot_height),
        ("slot_depth", geo.slot_depth),
        ("tip_radius", geo.tip_radius),
        ("body_length", geo.body_length),
        ("face_thickness", geo.face_thickness),
        ("face_od", geo.face_od),
    ):
        if value is not None and value <= 0.0:
            errs.append(f"{name} must be > 0")
    if geo.slot_length_over_dh <= 0.0:
        errs.append("slot_length_over_dh must be > 0")
    if spec.target_momentum_ratio is not None and spec.target_momentum_ratio <= 0:
        errs.append("target_momentum_ratio must be > 0")
    if spec.evaporation_constant <= 0.0:
        errs.append("evaporation_constant must be > 0")
    if int(spec.fuel_manifold_ports) < 1:
        errs.append("fuel_manifold_ports must be >= 1")
    if int(spec.oxidizer_manifold_ports) < 1:
        errs.append("oxidizer_manifold_ports must be >= 1")
    if spec.manufacturing.min_feature <= 0.0:
        errs.append(
            f"min_feature must be > 0, got {spec.manufacturing.min_feature}")
    for name, value in (
        ("web_min", spec.manufacturing.web_min),
        ("edge_distance_min", spec.manufacturing.edge_distance_min),
    ):
        if value is not None and value <= 0.0:
            errs.append(f"{name} must be > 0")
    if spec.manufacturing.concentricity_tolerance < 0.0:
        errs.append("concentricity_tolerance must be >= 0")
    if spec.sizing == "fixed":
        if geo.annulus_gap is None or geo.slot_width is None:
            errs.append(
                "fixed sizing requires both annulus_gap and slot_width "
                "(otherwise it would silently auto-size the geometry)")
    for feed in (spec.fuel, spec.oxidizer):
        if feed.phase not in ("auto", "liquid", "gas"):
            errs.append(f"{feed.role} phase must be auto, liquid, or gas")
        if feed.inlet_temperature is not None and feed.inlet_temperature <= 0.0:
            errs.append(f"{feed.role} inlet_temperature must be > 0")
        if feed.inlet_pressure is not None and feed.inlet_pressure <= 0.0:
            errs.append(f"{feed.role} inlet_pressure must be > 0")
        for name, value in (
            ("density", feed.density),
            ("viscosity", feed.viscosity),
            ("surface_tension", feed.surface_tension),
            ("vapor_pressure", feed.vapor_pressure),
        ):
            if value is not None and value <= 0.0:
                errs.append(f"{feed.role} {name} must be > 0")
    if mdot_fuel <= 0.0 or mdot_oxidizer <= 0.0:
        errs.append("fuel and oxidizer mass flows must be positive")
    if Pc <= 0.0:
        errs.append("chamber pressure must be positive")
    if mixture_ratio <= 0.0:
        errs.append("mixture_ratio must be positive")
    if chamber_radius <= 0.0:
        errs.append("chamber_radius must be positive")
    if chamber_length <= 0.0:
        errs.append("chamber_length must be positive")
    if gamma <= 1.0:
        errs.append("gamma must be greater than one")
    if Tc <= 0.0 or R_gas <= 0.0:
        errs.append("Tc and R_gas must be positive")
    if errs:
        raise InjectorSpecError("invalid injector spec: " + "; ".join(errs))


# ---------------------------------------------------------------------------
# Propellant feed-property resolution
# ---------------------------------------------------------------------------
# CoolProp HEOS covers these fluids directly (literature-grade equations of
# state; Bell et al., Ind. Eng. Chem. Res. 53 (2014)).
_COOLPROP_FLUIDS = {
    "oxygen": "Oxygen", "lox": "Oxygen", "o2": "Oxygen", "gox": "Oxygen",
    "lo2": "Oxygen",
    "methane": "Methane", "ch4": "Methane", "lch4": "Methane",
    "lng": "Methane",
    "hydrogen": "Hydrogen", "h2": "Hydrogen", "lh2": "Hydrogen",
    "gh2": "Hydrogen",
    "ethanol": "Ethanol", "etoh": "Ethanol",
    "water": "Water", "h2o": "Water",
    "nitrousoxide": "NitrousOxide", "n2o": "NitrousOxide",
    "nitrous": "NitrousOxide",
}

# Per-CoolProp-fluid default inlet temperature [K] when none is supplied
# (representative subcooled/storage states for liquid feed).
_DEFAULT_INLET_T = {
    "Oxygen": 90.0, "Methane": 112.0, "Hydrogen": 21.0,
    "Ethanol": 298.0, "Water": 298.0, "NitrousOxide": 270.0,
}

# Fluids CoolProp does not cover well: constant-property literature data,
# explicitly screening-grade (no T-dependence modelled).  Sutton & Biblarz
# RPE; NASA SP-8087; CRC handbook.
_LITERATURE_FEED_PROPERTIES = {
    "rp1": dict(
        rho=810.0, mu=1.6e-3, sigma=0.023, Pvap=2.0e3, T_ref=298.0,
        Tcrit=678.0, Pcrit=2.2e6, cp=2010.0, k=0.13,
        source="RP-1/Jet-A class (Sutton & Biblarz RPE; NASA SP-8087): "
        "rho~810 kg/m^3, mu~1.6e-3 Pa.s, sigma~0.023 N/m, Pvap~2 kPa @298 K "
        "(constant-property screening)",
    ),
    "mmh": dict(
        rho=874.0, mu=0.775e-3, sigma=0.0341, Pvap=6.6e3, T_ref=298.0,
        Tcrit=585.0, Pcrit=8.2e6, cp=2930.0, k=0.25,
        source="MMH (Sutton & Biblarz RPE; CRC): rho 874, mu 0.78e-3, "
        "sigma 0.034, Pvap 6.6 kPa @298 K (constant-property screening)",
    ),
    "n2o4": dict(
        rho=1443.0, mu=0.42e-3, sigma=0.0267, Pvap=96.0e3, T_ref=293.0,
        Tcrit=431.0, Pcrit=10.1e6, cp=1550.0, k=0.13,
        source="N2O4/NTO (Sutton & Biblarz RPE): rho 1443, mu 0.42e-3, "
        "sigma 0.0267, Pvap 96 kPa @293 K (volatile; low ambient cavitation "
        "margin; constant-property screening)",
    ),
    "udmh": dict(
        rho=791.0, mu=0.492e-3, sigma=0.0289, Pvap=16.3e3, T_ref=298.0,
        Tcrit=523.0, Pcrit=5.4e6, cp=2730.0, k=0.16,
        source="UDMH (Sutton & Biblarz RPE; CRC): rho 791, mu 0.49e-3, "
        "sigma 0.029, Pvap 16 kPa @298 K (constant-property screening)",
    ),
}

_LITERATURE_ALIASES = {
    "rp1": "rp1", "rp-1": "rp1", "kerosene": "rp1", "jeta": "rp1",
    "jet-a": "rp1", "jp8": "rp1",
    "mmh": "mmh", "monomethylhydrazine": "mmh",
    "n2o4": "n2o4", "nto": "n2o4", "mon": "n2o4", "mon3": "n2o4",
    "udmh": "udmh", "aerozine": "udmh",
}


def _norm(name: str | None) -> str:
    return (name or "").strip().lower().replace(" ", "").replace("_", "")


def resolve_feed_state(
    spec: PropellantFeedSpec,
    *,
    default_pressure: float,
    subcool_margin: float = 0.10,
) -> FeedState:
    """Resolve liquid feed properties (rho, mu, sigma, Pvap, phase).

    CoolProp is used for the cryogens / storable solvents it covers; a small
    literature table covers RP-1, MMH, N2O4 and UDMH.  Explicit overrides on
    the spec always win.  ``liquid_ok`` is False (with a reason) when the
    resolved state is gaseous, supercritical, or within ``subcool_margin`` of
    the vapor pressure (cavitation/flashing risk) — the liquid/liquid MVP
    cannot size those.
    """
    role = spec.role
    raw = spec.name or "(unnamed)"
    key = _norm(spec.name)
    P = float(spec.inlet_pressure if spec.inlet_pressure is not None
              else default_pressure)

    # 1) Full manual override.
    if all(v is not None for v in (
        spec.density, spec.viscosity, spec.surface_tension, spec.vapor_pressure
    )):
        T = float(spec.inlet_temperature or 298.0)
        phase = spec.phase if spec.phase in ("liquid", "gas") else "liquid"
        liquid_ok = phase == "liquid"
        return FeedState(
            role=role, name=raw, temperature=T, pressure=P,
            density=float(spec.density), viscosity=float(spec.viscosity),
            surface_tension=float(spec.surface_tension),
            vapor_pressure=float(spec.vapor_pressure), phase=phase,
            critical_temperature=None, critical_pressure=None,
            source=spec.property_source or "user-supplied property overrides",
            liquid_ok=liquid_ok,
            reason="" if liquid_ok else "user forced non-liquid phase",
        )

    # 2) CoolProp-covered fluid.
    fluid = _COOLPROP_FLUIDS.get(key)
    if fluid is not None:
        state = _coolprop_feed_state(spec, fluid, P, subcool_margin)
        if state is not None:
            return state
        # CoolProp unavailable -> fall through to literature if we have it.

    # 3) Literature constants.
    lit_key = _LITERATURE_ALIASES.get(key)
    if lit_key is not None:
        d = _LITERATURE_FEED_PROPERTIES[lit_key]
        T = float(spec.inlet_temperature if spec.inlet_temperature is not None
                  else d["T_ref"])
        Pvap = float(spec.vapor_pressure if spec.vapor_pressure is not None
                     else d["Pvap"])
        Tcrit = d.get("Tcrit")
        liquid_ok, gas_ok, phase, reason = _classify_phase(
            T, P, Pvap, Tcrit, spec.phase, subcool_margin
        )
        if gas_ok:
            # The literature table carries only liquid constants; it cannot
            # supply gas gamma/R, so a gas state here is not sizeable.
            gas_ok = False
            reason = (reason + "; literature table has no gas gamma/R "
                      "(use a CoolProp fluid or explicit overrides for gas)")
        return FeedState(
            role=role, name=raw, temperature=T, pressure=P,
            density=float(spec.density if spec.density is not None else d["rho"]),
            viscosity=float(spec.viscosity if spec.viscosity is not None
                            else d["mu"]),
            surface_tension=float(spec.surface_tension
                                  if spec.surface_tension is not None
                                  else d["sigma"]),
            vapor_pressure=Pvap, phase=phase,
            critical_temperature=Tcrit, critical_pressure=d.get("Pcrit"),
            source=spec.property_source or d["source"], liquid_ok=liquid_ok,
            reason=reason, gas_ok=gas_ok,
            cp=d.get("cp"), conductivity=d.get("k"),
        )

    raise InjectorUnsupportedState(
        f"unknown propellant '{raw}' for the {role} feed; supply explicit "
        f"density/viscosity/surface_tension/vapor_pressure overrides, or use "
        f"one of: {sorted(set(_COOLPROP_FLUIDS) | set(_LITERATURE_ALIASES))}"
    )


def _coolprop_feed_state(spec, fluid, P, subcool_margin) -> FeedState | None:
    try:
        from CoolProp.CoolProp import PropsSI
    except Exception:
        return None
    Tcrit = float(PropsSI("Tcrit", fluid))
    Pcrit = float(PropsSI("Pcrit", fluid))
    T = spec.inlet_temperature
    if T is None:
        T = _DEFAULT_INLET_T.get(fluid, 298.0)
    T = float(T)
    try:
        rho = float(PropsSI("Dmass", "T", T, "P", P, fluid))
        mu = float(PropsSI("VISCOSITY", "T", T, "P", P, fluid))
    except Exception as exc:
        raise InjectorUnsupportedState(
            f"CoolProp could not evaluate {fluid} at T={T:.1f} K, "
            f"P={P/1e5:.1f} bar: {exc}"
        ) from exc
    try:
        cp = float(PropsSI("Cpmass", "T", T, "P", P, fluid))
        k_th = float(PropsSI("CONDUCTIVITY", "T", T, "P", P, fluid))
    except Exception:
        cp = k_th = None
    # Saturation properties (only meaningful below the critical point).
    if T < Tcrit:
        try:
            Pvap = float(PropsSI("P", "T", T, "Q", 0, fluid))
            sigma = float(PropsSI("SURFACE_TENSION", "T", T, "Q", 0, fluid))
        except Exception:
            Pvap, sigma = float("nan"), float("nan")
    else:
        Pvap, sigma = float("nan"), float("nan")
    liquid_ok, gas_ok, phase, reason = _classify_phase(
        T, P, Pvap, Tcrit, spec.phase, subcool_margin
    )
    # Override surface tension if supplied (needed when supercritical etc.).
    if spec.surface_tension is not None:
        sigma = float(spec.surface_tension)
    # Real-gas gamma + specific gas constant for the compressible branch.
    gamma = gas_constant = None
    if gas_ok:
        try:
            cp = float(PropsSI("Cpmass", "T", T, "P", P, fluid))
            cv = float(PropsSI("Cvmass", "T", T, "P", P, fluid))
            mw = float(PropsSI("M", fluid))   # kg/mol
            gamma = cp / cv if cv > 0 else None
            gas_constant = 8.31446 / mw if mw > 0 else None
        except Exception:
            gamma = gas_constant = None
        if gamma is None or gas_constant is None:
            gas_ok = False
            reason = (reason + "; could not evaluate gas gamma/R for the "
                      "compressible branch")
    return FeedState(
        role=spec.role, name=spec.name or fluid, temperature=T, pressure=P,
        density=float(spec.density) if spec.density is not None else rho,
        viscosity=float(spec.viscosity) if spec.viscosity is not None else mu,
        surface_tension=sigma,
        vapor_pressure=float(spec.vapor_pressure)
        if spec.vapor_pressure is not None else Pvap,
        phase=phase, critical_temperature=Tcrit, critical_pressure=Pcrit,
        source=spec.property_source
        or f"CoolProp HEOS ({fluid}); Bell et al. IECR 53 (2014)",
        liquid_ok=liquid_ok, reason=reason, gas_ok=gas_ok,
        gamma=gamma, gas_constant=gas_constant, cp=cp, conductivity=k_th,
    )


def _classify_phase(T, P, Pvap, Tcrit, requested, subcool_margin):
    """Classify a feed state -> (liquid_ok, gas_ok, phase, reason).

    * liquid_ok  -> incompressible orifice branch.
    * gas_ok     -> compressible/choked orifice branch (gas or supercritical
      dense gas; real-fluid screening).
    * neither    -> a two-phase / flashing state within ``subcool_margin`` of
      the vapor pressure, which neither branch can size.
    """
    if requested == "gas":
        return False, True, "gas", "phase forced to gas (compressible branch)"
    if Tcrit is not None and T >= 0.98 * Tcrit:
        return False, True, "supercritical", (
            f"T={T:.1f} K at/above 0.98*Tcrit={0.98*Tcrit:.1f} K -> "
            f"compressible (dense-gas) branch")
    if not math.isnan(Pvap):
        if P <= Pvap:
            if requested == "liquid":
                return False, False, "gas", (
                    f"phase forced to liquid but feed pressure {P/1e5:.2f} bar "
                    f"<= vapor pressure {Pvap/1e5:.2f} bar")
            return False, True, "gas", (
                f"feed pressure {P/1e5:.2f} bar <= vapor pressure "
                f"{Pvap/1e5:.2f} bar -> compressible (gas) branch")
        if P < (1.0 + subcool_margin) * Pvap:
            return False, False, "two_phase", (
                f"feed pressure {P/1e5:.2f} bar within {subcool_margin*100:.0f}% "
                f"of vapor pressure {Pvap/1e5:.2f} bar (cavitation/flashing "
                f"risk; neither liquid nor gas branch applies)")
    return True, False, "liquid", ""


# ---------------------------------------------------------------------------
# Hydraulic sizing
# ---------------------------------------------------------------------------
def _annulus_from_area(area: float, pintle_diameter: float) -> dict:
    """Annular gap geometry from a required flow area and inner diameter."""
    Di = pintle_diameter
    Do = math.sqrt(Di * Di + 4.0 * area / math.pi)
    gap = 0.5 * (Do - Di)
    return {"inner_diameter": Di, "outer_diameter": Do, "gap": gap,
            "hydraulic_diameter": Do - Di}


def _slots_from_area(area: float, n_slots: int, aspect_ratio: float,
                     pintle_diameter: float, slot_depth: float | None,
                     length_over_dh: float) -> dict:
    """Rectangular-slot geometry dividing a required radial flow area."""
    a_each = area / n_slots
    w = math.sqrt(a_each / aspect_ratio)
    h = aspect_ratio * w
    dh = 2.0 * w * h / (w + h)
    depth = slot_depth if slot_depth is not None else max(
        length_over_dh * dh, 0.0
    )
    circumference = math.pi * pintle_diameter
    web = circumference / n_slots - w
    return {
        "slot_width": w, "slot_height": h, "slot_depth": depth,
        "hydraulic_diameter": dh, "web": web,
        "blockage_factor": n_slots * w / circumference,
        "length_over_dh": depth / dh if dh > 0 else float("nan"),
        "area_each": a_each,
    }


def _stream_numbers(role, geom, mdot, dp, cd, area, dh, rho, mu, sigma,
                    velocity=None):
    v = velocity if velocity is not None else mdot / (rho * area)
    re = rho * v * dh / mu if mu > 0 else float("nan")
    we = rho * v * v * dh / sigma if sigma > 0 else float("nan")
    oh = mu / math.sqrt(rho * sigma * dh) if sigma > 0 else float("nan")
    return StreamResult(
        role=role, geometry=geom, mdot=mdot, dp=dp, cd=cd, area=area,
        velocity=v, hydraulic_diameter=dh, reynolds=re, weber=we, ohnesorge=oh,
    )


def _stream_mass_flux(fs: FeedState, dp: float, cd: float, P_back: float):
    """Mass flux ``G = mdot/A`` [kg/(m^2 s)] for one feed stream.

    Liquid -> incompressible orifice ``G = Cd sqrt(2 rho dp)``.
    Gas / supercritical -> compressible orifice with an explicit choke test
    against the critical pressure ratio (Sutton & Biblarz; Anderson, *Modern
    Compressible Flow*).  Returns ``(G, v_inj, choked, branch, info)`` where
    ``v_inj`` is the throat injection velocity (sonic when choked).
    """
    if fs.liquid_ok:
        G = cd * math.sqrt(2.0 * fs.density * dp)
        return G, G / fs.density, False, "incompressible", {}
    g, R, T0 = fs.gamma, fs.gas_constant, fs.temperature
    if not (g and g > 1.0 and R and R > 0.0 and T0 and T0 > 0.0):
        raise InjectorUnsupportedState(
            f"{fs.role} gas feed lacks gamma/R for the compressible branch")
    P0 = P_back + dp                      # upstream stagnation pressure
    crit = (2.0 / (g + 1.0)) ** (g / (g - 1.0))
    pr = P_back / P0
    info = {"critical_pressure_ratio": crit, "pressure_ratio": pr,
            "stagnation_pressure": P0}
    if pr <= crit:                        # choked
        G = (cd * P0 * math.sqrt(g / (R * T0))
             * (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0))))
        v = math.sqrt(g * R * T0 * 2.0 / (g + 1.0))   # sonic at the throat
        return G, v, True, "compressible_choked", info
    G = cd * P0 * math.sqrt(
        2.0 * g / ((g - 1.0) * R * T0)
        * (pr ** (2.0 / g) - pr ** ((g + 1.0) / g)))
    M = math.sqrt(max(0.0, 2.0 / (g - 1.0)
                      * ((P0 / P_back) ** ((g - 1.0) / g) - 1.0)))
    T = T0 / (1.0 + (g - 1.0) / 2.0 * M * M)
    v = M * math.sqrt(g * R * T)
    info["exit_mach"] = M
    return G, v, False, "compressible_subsonic", info


# Atomization / vaporization screening constants (all clearly screening-grade).
_HINZE_CRITICAL_WEBER = 13.0      # max stable drop We_g (Hinze 1955)
_PRIMARY_BREAKUP_DIAMETERS = 15.0  # atomization-regime primary breakup length / d_jet
_ATOMIZATION_WEBER_FLOOR = 40.0   # below this We_g, primary breakup is poor
_DEFAULT_EVAPORATION_CONSTANT = 1.0e-6   # m^2/s, d^2-law burning-rate K (hydrocarbon class)


def _stream_atomization(s, feed, rho_g, chamber_length, K_b):
    """Primary-atomization + d^2-law vaporization screen for one stream.

    SMD from the Hinze critical-Weber aerodynamic-breakup limit
    (``d_32 = We_crit sigma / (rho_g v^2)``, Hinze, AIChE J. 1 (1955)),
    capped at the jet hydraulic diameter.  In the atomization regime primary
    breakup completes within ~10-30 jet diameters (Reitz & Bracco), taken here
    as ``L_b = C d_jet``.  Vaporization follows the d^2-law
    (``d^2(t) = d_32^2 - K_b t``) over the post-breakup residence, the chamber
    residence using the injection velocity as the convective scale.  Combustion
    efficiency is approximated by the vaporized mass fraction (Priem & Heidmann,
    NASA TR R-67, 1960: vaporization-limited combustion).  The breakup length and
    the vaporized fraction share the same residence so they stay consistent.
    """
    v = s.velocity
    d_jet = s.hydraulic_diameter
    sigma = feed.surface_tension
    rho_l = feed.density
    if not (sigma > 0) or not (rho_g > 0) or not (v > 0):
        return StreamAtomization(
            role=s.role, sauter_mean_diameter=float("nan"),
            aerodynamic_weber=float("nan"), breakup_length=float("nan"),
            vaporization_length=float("nan"), combustion_length=float("nan"),
            vaporized_fraction=float("nan"),
            regime="indeterminate (missing surface tension / gas density)")
    we_g = rho_g * v * v * d_jet / sigma
    d32 = min(_HINZE_CRITICAL_WEBER * sigma / (rho_g * v * v), d_jet)
    L_breakup = _PRIMARY_BREAKUP_DIAMETERS * d_jet
    # length to ~99% vaporized mass (1% volume remaining → d=d32·0.01^(1/3))
    t_99 = d32 * d32 * (1.0 - 0.01 ** (2.0 / 3.0)) / K_b
    L_vap = v * t_99
    L_comb = L_breakup + L_vap
    # vaporized fraction in the chamber length downstream of primary breakup
    t_res = max(0.0, chamber_length - L_breakup) / v
    d_rem2 = max(0.0, d32 * d32 - K_b * t_res)
    vap_frac = 1.0 - (d_rem2 / (d32 * d32)) ** 1.5 if d32 > 0 else 1.0
    regime = ("aerodynamic atomization" if we_g >= _ATOMIZATION_WEBER_FLOOR
              else f"below atomization regime (We_g={we_g:.0f} < "
                   f"{_ATOMIZATION_WEBER_FLOOR:.0f}; poor primary breakup, "
                   f"cold-flow required)")
    return StreamAtomization(
        role=s.role, sauter_mean_diameter=d32, aerodynamic_weber=we_g,
        breakup_length=L_breakup, vaporization_length=L_vap,
        combustion_length=L_comb, vaporized_fraction=vap_frac, regime=regime)


def spray_atomization(
    streams: dict, feed: dict, *, Pc, Tc, R_gas, chamber_length,
    evaporation_constant: float = _DEFAULT_EVAPORATION_CONSTANT,
) -> SprayAtomization:
    """Whole-injector spray atomization / vaporization / c* screen."""
    rho_g = Pc / (R_gas * Tc)
    per = {
        role: _stream_atomization(streams[role], feed[role], rho_g,
                                  chamber_length, evaporation_constant)
        for role in ("fuel", "oxidizer")
    }
    # The limiting (longest-combustion-length) stream sets the development need.
    limiting_role = max(
        per, key=lambda r: (per[r].combustion_length
                            if per[r].combustion_length == per[r].combustion_length
                            else -1.0))
    L_comb = per[limiting_role].combustion_length
    margin = chamber_length / L_comb if L_comb > 0 else float("inf")
    # Mass-weighted vaporized fraction → c* efficiency surrogate.
    mdot = {r: streams[r].mdot for r in per}
    total = sum(mdot.values())
    eta = sum(per[r].vaporized_fraction * mdot[r] for r in per) / max(total, 1e-12)
    notes = [
        "Order-of-magnitude screening only (Hinze SMD + Reitz-Bracco breakup + "
        "d^2-law + Priem-Heidmann vaporization-limited c*); spray distribution "
        "and SMD require cold-flow validation (NASA SP-8089).",
    ]
    return SprayAtomization(
        chamber_gas_density=rho_g, evaporation_constant=evaporation_constant,
        streams=per, limiting_role=limiting_role, combustion_length=L_comb,
        available_chamber_length=chamber_length, development_margin=margin,
        predicted_cstar_efficiency=float(min(max(eta, 0.0), 1.0)),
        model="hinze_reitzbracco_d2law_priem_heidmann_screen", notes=notes)


def manifold_distribution(result, spec, dp_fuel, dp_ox) -> ManifoldDistribution:
    """Per-propellant manifold maldistribution (annular two-header network).

    Each propellant manifold is an annular header ring fed by discrete ports;
    the slotted stream distributes into the slots and the annulus into a
    circumferential discretization.  The square-law network gives the
    element-to-element flow spread for the assumed geometry (NASA SP-8087;
    Kang & Sun, JTHT 25, 2011).  Gas streams use the same incompressible
    network as a screen.
    """
    from raosim.thermofluids import solve_annular_manifold_network
    dp_by_role = {"fuel": dp_fuel, "oxidizer": dp_ox}
    ports_by_role = {"fuel": int(spec.fuel_manifold_ports),
                     "oxidizer": int(spec.oxidizer_manifold_ports)}
    streams: dict[str, ManifoldResult] = {}
    worst = 0.0
    for role, s in result.streams.items():
        feeds = s.geometry
        dp = dp_by_role[role]
        ports = max(ports_by_role[role], 1)
        if feeds == "slots":
            n_elem = max(int(result.slot_count), 2)
            manifold_radius = 0.5 * result.pintle_diameter
        else:
            n_elem = max(12, 4 * ports)   # discretize the continuous annulus
            manifold_radius = 0.5 * s.detail.get(
                "outer_diameter", result.pintle_diameter)
        try:
            net = solve_annular_manifold_network(
                channel_count=n_elem, ports_per_manifold=ports,
                total_mass_flow=s.mdot, density=result.feed[role].density,
                channel_pressure_drop=max(dp, 1.0), channel_total_area=s.area,
                manifold_radius=manifold_radius)
            spread = float(net["maldistribution_fraction"])
            mr = ManifoldResult(
                role=role, feeds=feeds, element_count=n_elem, port_count=ports,
                maldistribution_fraction=spread,
                min_flow_ratio=float(net["minimum_channel_flow_ratio"]),
                max_flow_ratio=float(net["maximum_channel_flow_ratio"]),
                manifold_pressure_drop=float(
                    net["total_pressure_drop"]
                    - net["channel_branch_pressure_drop_reference"]),
                port_diameter=float(net["port_diameter"]),
                plenum_area=float(net["plenum_cross_section_area"]),
                status=str(net["status"]))
            if spread == spread:
                worst = max(worst, spread)
        except Exception as exc:
            mr = ManifoldResult(
                role=role, feeds=feeds, element_count=n_elem, port_count=ports,
                maldistribution_fraction=float("nan"),
                min_flow_ratio=float("nan"), max_flow_ratio=float("nan"),
                manifold_pressure_drop=float("nan"),
                port_diameter=float("nan"), plenum_area=float("nan"),
                status=f"unscreened:{type(exc).__name__}")
        streams[role] = mr
    return ManifoldDistribution(
        streams=streams, worst_maldistribution=worst,
        notes=["1-D annular two-header square-law network (NASA SP-8087; "
               "Kang & Sun JTHT 25, 2011); requires 3-D manifold validation."])


def _convective_wall(T_aw, T_cool, h_g, h_c, t_wall, k_wall):
    """1-D series gas/wall/coolant circuit -> (q, T_wg)."""
    R = 1.0 / max(h_g, 1e-12) + t_wall / max(k_wall, 1e-9) + 1.0 / max(h_c, 1e-12)
    q = (T_aw - T_cool) / R
    T_wg = T_aw - q / max(h_g, 1e-12)
    return q, T_wg


def face_tip_thermal(result, spec, *, Pc, Tc, gamma, R_gas,
                     recirculation_temp_fraction=0.8,
                     recirculation_velocity_fraction=0.2,
                     tip_wall_thickness=None):
    """Screening heat balance for the injector face and pintle tip.

    Both surfaces sit in recirculating combustion gas (recovery temperature
    ~``f·Tc``) and are cooled from behind by the propellant passing through the
    pintle (tip) and the manifolds (face).  A turbulent Dittus-Boelter gas-side
    coefficient and a Dittus-Boelter coolant-side coefficient feed a 1-D
    series gas/wall/coolant circuit (cf. the SP-125 regen wall solve), giving a
    gas-side wall temperature and a margin against the material limit.  This is
    a screening indicator (recirculation/film effects need CFD/cold-flow), not
    a qualified thermal analysis.
    """
    from types import SimpleNamespace
    from raosim.physics import gas_transport_properties
    notes = []
    # Chamber-gas transport + density.
    ns = SimpleNamespace(gamma=gamma, R_gas=R_gas,
                         Mw=8.31446 / R_gas, Tc=Tc)
    try:
        cp_g, Pr_g, mu_g = gas_transport_properties(ns)
    except Exception:
        cp_g, Pr_g, mu_g = 2000.0, 0.5, 1.0e-4
        notes.append("gas transport properties estimated (fallback)")
    rho_g = Pc / (R_gas * Tc)
    k_g = mu_g * cp_g / max(Pr_g, 1e-6)
    T_aw = recirculation_temp_fraction * Tc

    # Pintle/face material limit + conductivity.
    k_wall, T_limit = 350.0, 800.0
    try:
        from raosim.materials import get_material
        if spec.pintle_material:
            mat = get_material(spec.pintle_material)
            k_wall, T_limit = mat.conductivity, mat.max_temperature
    except Exception:
        notes.append("pintle material unresolved; copper-class defaults used")
    t_wall = (tip_wall_thickness if tip_wall_thickness is not None
              else max(2.0 * spec.manufacturing.min_feature, 1.0e-3))

    def _surface(stream, length, label):
        fs = result.feed[stream.role]
        # gas-side recirculation Dittus-Boelter
        U = recirculation_velocity_fraction * stream.velocity
        Re_g = rho_g * U * length / max(mu_g, 1e-12)
        h_g = 0.023 * Re_g ** 0.8 * Pr_g ** 0.4 * k_g / max(length, 1e-9)
        # coolant-side Dittus-Boelter (propellant through the passage)
        cp_l = fs.cp if fs.cp else 2000.0
        k_l = fs.conductivity if fs.conductivity else 0.13
        if fs.cp is None or fs.conductivity is None:
            notes.append(f"{stream.role} cp/k estimated (fallback)")
        Pr_l = fs.viscosity * cp_l / max(k_l, 1e-9)
        dh = stream.hydraulic_diameter
        h_c = (0.023 * max(stream.reynolds, 1.0) ** 0.8 * Pr_l ** 0.4
               * k_l / max(dh, 1e-9))
        q, T_wg = _convective_wall(T_aw, fs.temperature, h_g, h_c, t_wall, k_wall)
        return {"h_g": h_g, "h_c": h_c, "q": q, "T_wg": T_wg,
                "margin": T_limit / max(T_wg, 1.0), "label": label}

    # Tip cooled by the radial (slotted) stream; face by the annulus stream.
    tip = _surface(result.slots, result.pintle_diameter, "tip")
    face = _surface(result.annulus, result.chamber_radius, "face")
    limiting = "tip" if tip["margin"] <= face["margin"] else "face"
    return FaceTipThermal(
        recovery_temperature=T_aw,
        tip_gas_coefficient=tip["h_g"], tip_coolant_coefficient=tip["h_c"],
        tip_heat_flux=tip["q"], tip_wall_temperature=tip["T_wg"],
        tip_margin=tip["margin"], face_heat_flux=face["q"],
        face_wall_temperature=face["T_wg"], face_margin=face["margin"],
        limiting=limiting, wall_temperature_limit=T_limit,
        governing_margin=min(tip["margin"], face["margin"]),
        model="recirculation_dittus_boelter_series_circuit_screen",
        notes=notes)


def stability_screen(result, *, Pc, Tc, gamma, R_gas, chi_fuel, chi_oxidizer):
    """Combustion-stability screen beyond the bare ``c/2L`` estimate.

    Three coupled mechanisms, all screening-grade (NASA SP-8113 / SP-194,
    *Liquid Rocket Engine Combustion Stability*; Crocco & Cheng sensitive
    time-lag n-τ):

    * **Chug / feed-system coupling** — the injector pressure-drop fraction
      decouples the feed from the chamber; ``min(χ_f, χ_o) ≥ ~0.2`` is the
      usual stability rule, ``< 0.1`` is chug-prone.
    * **Chamber acoustics** — longitudinal ``a/(2L)`` and the transverse
      tangential ``1.8412·a/(πD)`` and radial ``3.8317·a/(πD)`` modes.
    * **n-τ intrinsic coupling** — the combustion time lag ``τ`` (taken as the
      atomization/vaporization development time) against the L1 period; the
      reduced frequency ``τ·f_L1`` falling in the first sensitive band flags a
      high-frequency screening concern.
    """
    a = math.sqrt(gamma * R_gas * Tc)
    Lc = max(result.chamber_length, 1e-9)
    Dc = max(2.0 * result.chamber_radius, 1e-9)
    f_L1 = a / (2.0 * Lc)
    f_L2 = 2.0 * f_L1
    f_T1 = 1.8412 * a / (math.pi * Dc)
    f_R1 = 3.8317 * a / (math.pi * Dc)
    decoupling = min(chi_fuel, chi_oxidizer)
    chug = ("good (≥0.2 Pc)" if decoupling >= 0.20 else
            "marginal (0.1–0.2 Pc)" if decoupling >= 0.10 else
            "chug-prone (<0.1 Pc)")
    # Combustion time lag ~ development length / limiting injection velocity.
    if result.atomization is not None:
        lim = result.atomization.limiting_role
        v = max(result.streams[lim].velocity, 1e-6)
        tau = result.atomization.combustion_length / v
    else:
        tau = float("nan")
    reduced = tau * f_L1 if tau == tau else float("nan")
    sensitive = bool(0.1 < reduced < 0.5) if reduced == reduced else False
    notes = [
        "Screening only: chug rule (SP-8113/SP-194), closed-chamber acoustic "
        "modes, and an n-τ reduced-frequency band (Crocco sensitive time lag). "
        "A real stability assessment needs the feed admittance, the combustion "
        "response function, and damping (baffles/cavities).",
    ]
    return StabilityScreen(
        sound_speed=a, f_L1=f_L1, f_L2=f_L2, f_T1=f_T1, f_R1=f_R1,
        injector_decoupling_fraction=decoupling, chug_status=chug,
        combustion_time_lag=tau, reduced_frequency_L1=reduced,
        sensitive_band=sensitive,
        model="chug_plus_chamber_acoustics_plus_ntau_screen", notes=notes)


@dataclass
class ThrottlePoint:
    throttle: float                 # mdot/mdot_full
    Pc: float
    mdot_total: float
    mixture_ratio: float
    dp_fuel_fraction: float
    dp_oxidizer_fraction: float
    annulus_gap: float
    slot_width: float
    sleeve_stroke_fraction: float   # annulus area(f) / area(full)
    v_annulus: float
    v_slots: float
    reynolds_slots: float
    weber_slots: float
    total_momentum_ratio: float
    spray_half_angle_deg: float
    spray_wall_axial_distance: float
    smd_limiting: float
    predicted_cstar_efficiency: float
    thermal_margin: float
    feasible: bool


@dataclass
class ThrottleMap:
    points: list[ThrottlePoint]
    preserved: dict[str, bool]
    pc_exponent: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pc_exponent": self.pc_exponent,
            "preserved": self.preserved,
            "points": [
                {
                    "throttle": p.throttle, "Pc_pa": p.Pc,
                    "mdot_total_kg_s": p.mdot_total,
                    "mixture_ratio": p.mixture_ratio,
                    "dp_fuel_fraction": p.dp_fuel_fraction,
                    "dp_oxidizer_fraction": p.dp_oxidizer_fraction,
                    "annulus_gap_m": p.annulus_gap, "slot_width_m": p.slot_width,
                    "sleeve_stroke_fraction": p.sleeve_stroke_fraction,
                    "v_annulus_m_s": p.v_annulus, "v_slots_m_s": p.v_slots,
                    "reynolds_slots": p.reynolds_slots,
                    "weber_slots": p.weber_slots,
                    "total_momentum_ratio": p.total_momentum_ratio,
                    "spray_half_angle_deg": p.spray_half_angle_deg,
                    "spray_wall_axial_distance_m": p.spray_wall_axial_distance,
                    "smd_limiting_m": p.smd_limiting,
                    "predicted_cstar_efficiency": p.predicted_cstar_efficiency,
                    "thermal_margin": p.thermal_margin, "feasible": p.feasible,
                }
                for p in self.points
            ],
            "notes": self.notes,
        }


def throttle_map(
    spec: InjectorSpec, *, mdot_fuel_full, mdot_oxidizer_full, Pc_full,
    mixture_ratio, chamber_radius, chamber_length, gamma, Tc, R_gas,
    levels=(0.2, 0.4, 0.6, 0.8, 1.0), pc_exponent=1.0,
) -> ThrottleMap:
    """Movable-sleeve throttle schedule: resize the area to hold the
    pressure-drop fractions (and therefore O/F and TMR) constant as the engine
    throttles, instead of letting a fixed area collapse ΔP ∝ ṁ².

    The chamber pressure follows ``Pc(f) = Pc_full · f^pc_exponent`` (1.0 = the
    physical deep-throttle case Pc ∝ ṁ; 0.0 = a constant-Pc study).  At each
    level the auto solver re-sizes the openings to the dp-fraction, which the
    movable sleeve realizes as a stroke schedule (the annulus area ratio).
    """
    pts: list[ThrottlePoint] = []
    results: list[tuple[float, InjectorDesignResult]] = []
    for f in sorted(levels):
        if not (0.0 < f <= 1.0):
            raise InjectorSpecError("throttle levels must be in (0, 1]")
        Pc_f = Pc_full * (f ** pc_exponent)
        local = copy.deepcopy(spec)
        local.sizing = "auto"   # the movable sleeve resizes to the dp-fraction
        r = size_pintle_injector(
            local, mdot_fuel=f * mdot_fuel_full,
            mdot_oxidizer=f * mdot_oxidizer_full, Pc=Pc_f,
            mixture_ratio=mixture_ratio, chamber_radius=chamber_radius,
            chamber_length=chamber_length, gamma=gamma, Tc=Tc, R_gas=R_gas)
        results.append((f, r))
    area_full = results[-1][1].annulus.area
    for f, r in results:
        at = r.atomization
        lim = at.streams[at.limiting_role].sauter_mean_diameter if at else float("nan")
        pts.append(ThrottlePoint(
            throttle=f, Pc=Pc_full * (f ** pc_exponent),
            mdot_total=r.annulus.mdot + r.slots.mdot,
            mixture_ratio=r.streams["oxidizer"].mdot / max(r.streams["fuel"].mdot, 1e-12),
            dp_fuel_fraction=spec.fuel_dp_fraction,
            dp_oxidizer_fraction=spec.oxidizer_dp_fraction,
            annulus_gap=r.annulus.detail["gap"],
            slot_width=r.slots.detail["slot_width"],
            sleeve_stroke_fraction=r.annulus.area / max(area_full, 1e-30),
            v_annulus=r.annulus.velocity, v_slots=r.slots.velocity,
            reynolds_slots=r.slots.reynolds, weber_slots=r.slots.weber,
            total_momentum_ratio=r.total_momentum_ratio,
            spray_half_angle_deg=r.spray_half_angle_deg,
            spray_wall_axial_distance=r.spray_wall_axial_distance,
            smd_limiting=lim,
            predicted_cstar_efficiency=(
                at.predicted_cstar_efficiency if at else float("nan")),
            thermal_margin=(
                r.thermal.governing_margin if r.thermal else float("nan")),
            feasible=r.feasible))

    def _spread(vals):
        vals = [v for v in vals if v == v]
        if not vals:
            return float("inf")
        m = sum(vals) / len(vals)
        return (max(vals) - min(vals)) / m if m else 0.0
    preserved = {
        # O/F is exact; TMR holds to ~1% because liquid density is weakly
        # pressure-dependent (the manifold pressure shifts with Pc).
        "mixture_ratio": _spread([p.mixture_ratio for p in pts]) < 1e-4,
        "dp_fraction": True,   # held by construction
        "total_momentum_ratio": _spread(
            [p.total_momentum_ratio for p in pts]) < 1e-2,
    }
    notes = [
        "Movable-sleeve schedule holding the dp-fractions constant; O/F and "
        "TMR are preserved while velocity/Re/We and atomization fall toward "
        "low throttle (deep-throttle reality). Stroke = annulus area ratio.",
    ]
    return ThrottleMap(points=pts, preserved=preserved,
                       pc_exponent=pc_exponent, notes=notes)


def size_pintle_injector(
    spec: InjectorSpec,
    *,
    mdot_fuel: float,
    mdot_oxidizer: float,
    Pc: float,
    mixture_ratio: float,
    chamber_radius: float,
    chamber_length: float,
    gamma: float,
    Tc: float,
    R_gas: float,
    feed: dict[str, FeedState] | None = None,
) -> InjectorDesignResult:
    """Size (auto) or evaluate (fixed) a liquid/liquid pintle injector.

    ``feed`` may be pre-resolved (e.g. the fuel taken from the regen outlet);
    otherwise it is resolved from ``spec.fuel`` / ``spec.oxidizer``.
    """
    _validate_injector_spec(
        spec, mdot_fuel, mdot_oxidizer, Pc, mixture_ratio,
        chamber_radius, chamber_length, gamma, Tc, R_gas,
    )
    radial = spec.geometry.radial_stream

    dp_fuel = spec.fuel_dp_fraction * Pc
    dp_ox = spec.oxidizer_dp_fraction * Pc
    p_manifold_fuel = Pc + dp_fuel
    p_manifold_ox = Pc + dp_ox

    if feed is None:
        feed = {
            "fuel": resolve_feed_state(
                spec.fuel, default_pressure=p_manifold_fuel),
            "oxidizer": resolve_feed_state(
                spec.oxidizer, default_pressure=p_manifold_ox),
        }

    # Phase guard: each stream must be a usable liquid (incompressible branch)
    # or a usable gas/supercritical state (compressible branch).  A two-phase /
    # flashing state near the vapor pressure is sizeable by neither.
    for role, fs in feed.items():
        if not (fs.liquid_ok or fs.gas_ok):
            raise InjectorUnsupportedState(
                f"{role} feed ('{fs.name}') is sizeable by neither the liquid "
                f"nor the gas branch: {fs.reason}."
            )

    streams_in = {
        "fuel": (mdot_fuel, dp_fuel, spec.fuel_cd, feed["fuel"]),
        "oxidizer": (mdot_oxidizer, dp_ox, spec.oxidizer_cd, feed["oxidizer"]),
    }
    axial = "oxidizer" if radial == "fuel" else "fuel"

    # Pintle diameter anchor.  In auto mode without an explicit diameter we
    # take a fraction of the chamber radius (a packaging default) so the
    # annulus and slots have a real geometric reference.
    Dp = spec.geometry.pintle_diameter
    if Dp is None:
        Dp = 0.30 * (2.0 * chamber_radius)

    # In "auto" sizing the area is solved from the required mass flow so the
    # delivered flow equals the requested flow exactly.  In "fixed" sizing the
    # supplied geometry sets the area and the DELIVERED flow is computed from
    # the orifice law (it may not match the cycle split — the closure gate then
    # reports the drift, evaluating the design rather than resizing it).
    fixed = spec.sizing == "fixed"

    # ----- axial annulus stream -----
    m_a_req, dp_a, cd_a, fs_a = streams_in[axial]
    G_a, v_a, choked_a, branch_a, info_a = _stream_mass_flux(fs_a, dp_a, cd_a, Pc)
    if fixed and spec.geometry.annulus_gap is not None:
        gap = spec.geometry.annulus_gap
        Do = Dp + 2.0 * gap
        A_a = math.pi / 4.0 * (Do * Do - Dp * Dp)
        ann_geom = {"inner_diameter": Dp, "outer_diameter": Do, "gap": gap,
                    "hydraulic_diameter": Do - Dp}
    else:
        A_a = m_a_req / G_a
        ann_geom = _annulus_from_area(A_a, Dp)
    # Annulus passage length: the injector face thickness when known, else the
    # same L/D target used for slots, so L/D is always reported.
    ann_len = (spec.geometry.face_thickness or spec.geometry.body_length
               or spec.geometry.slot_length_over_dh
               * ann_geom["hydraulic_diameter"])
    ann_geom["length_over_dh"] = (
        ann_len / ann_geom["hydraulic_diameter"]
        if ann_geom["hydraulic_diameter"] > 0 else float("nan"))
    ann_geom["area"] = A_a
    m_a = G_a * A_a  # delivered
    annulus = _stream_numbers(
        axial, "annulus", m_a, dp_a, cd_a, A_a, ann_geom["hydraulic_diameter"],
        fs_a.density, fs_a.viscosity, fs_a.surface_tension, velocity=v_a)
    ann_geom["injection"] = {"branch": branch_a, "choked": choked_a, **info_a}
    annulus.detail = ann_geom

    # ----- radial slot stream -----
    m_r_req, dp_r, cd_r, fs_r = streams_in[radial]
    G_r, v_r, choked_r, branch_r, info_r = _stream_mass_flux(fs_r, dp_r, cd_r, Pc)
    if fixed and spec.geometry.slot_width is not None:
        w = spec.geometry.slot_width
        h = spec.geometry.slot_height or (spec.geometry.slot_aspect_ratio * w)
        depth = spec.geometry.slot_depth
        dh = 2.0 * w * h / (w + h)
        circ = math.pi * Dp
        A_r = spec.geometry.slot_count * w * h
        slot_geom = {
            "slot_width": w, "slot_height": h,
            "slot_depth": depth if depth is not None else float("nan"),
            "hydraulic_diameter": dh, "web": circ / spec.geometry.slot_count - w,
            "blockage_factor": spec.geometry.slot_count * w / circ,
            "length_over_dh": (depth / dh) if depth else float("nan"),
            "area_each": w * h, "area": A_r,
        }
    else:
        A_r = m_r_req / G_r
        slot_geom = _slots_from_area(
            A_r, spec.geometry.slot_count, spec.geometry.slot_aspect_ratio,
            Dp, spec.geometry.slot_depth, spec.geometry.slot_length_over_dh)
        slot_geom["area"] = A_r
    m_r = G_r * A_r  # delivered
    slots = _stream_numbers(
        radial, "slots", m_r, dp_r, cd_r, A_r, slot_geom["hydraulic_diameter"],
        fs_r.density, fs_r.viscosity, fs_r.surface_tension, velocity=v_r)
    slot_geom["injection"] = {"branch": branch_r, "choked": choked_r, **info_r}
    slots.detail = slot_geom

    # Required (cycle) mass flows per role, for the closure gate.
    mdot_required = {"fuel": mdot_fuel, "oxidizer": mdot_oxidizer}

    # ----- momentum / spray -----
    tmr = (slots.mdot * slots.velocity) / (annulus.mdot * annulus.velocity)
    delta = math.radians(spec.geometry.deflector_angle)
    m_radial = slots.mdot * slots.velocity
    m_axial = annulus.mdot * annulus.velocity
    # Resultant spray direction from the radial/axial momentum vectors; the
    # deflector angle tilts the radial momentum toward the chamber axis.
    radial_comp = m_radial * math.cos(delta)
    axial_comp = m_axial + m_radial * math.sin(delta)
    spray_half_angle = math.degrees(math.atan2(radial_comp, axial_comp))
    spray_tan = math.tan(math.radians(spray_half_angle))
    # The spray cone originates at the pintle tip radius (tip_radius when
    # supplied, else the pintle radius) and the radial/axial streams interact
    # an impingement_distance downstream of the openings.
    r0 = spec.geometry.tip_radius if spec.geometry.tip_radius else 0.5 * Dp
    impingement = spec.geometry.impingement_distance or 0.0
    if 0.0 < spray_half_angle < 90.0 and spray_tan > 1e-6:
        wall_axial = impingement + (chamber_radius - r0) / spray_tan
    else:
        wall_axial = float("inf")

    width_ratio = slot_geom["slot_width"] / max(ann_geom["gap"], 1e-12)

    streams = {axial: annulus, radial: slots}
    result = InjectorDesignResult(
        feasible=True, sizing=spec.sizing, radial_stream=radial,
        pintle_diameter=Dp, slot_count=int(spec.geometry.slot_count),
        chamber_radius=chamber_radius,
        chamber_length=chamber_length, streams=streams, annulus=annulus,
        slots=slots, total_momentum_ratio=tmr,
        spray_half_angle_deg=spray_half_angle,
        spray_wall_axial_distance=wall_axial,
        slot_to_annulus_width_ratio=width_ratio,
        blockage_factor=slot_geom["blockage_factor"],
        minimum_web=slot_geom["web"], gates=[], feed=feed,
    )
    result.atomization = spray_atomization(
        streams, feed, Pc=Pc, Tc=Tc, R_gas=R_gas,
        chamber_length=chamber_length,
        evaporation_constant=spec.evaporation_constant)
    result.manifold = manifold_distribution(result, spec, dp_fuel, dp_ox)
    result.thermal = face_tip_thermal(
        result, spec, Pc=Pc, Tc=Tc, gamma=gamma, R_gas=R_gas)
    result.stability = stability_screen(
        result, Pc=Pc, Tc=Tc, gamma=gamma, R_gas=R_gas,
        chi_fuel=spec.fuel_dp_fraction, chi_oxidizer=spec.oxidizer_dp_fraction)
    result.gates = injector_gates(
        spec, result, Pc=Pc, mixture_ratio=mixture_ratio,
        dp_fuel=dp_fuel, dp_ox=dp_ox, p_manifold_fuel=p_manifold_fuel,
        p_manifold_ox=p_manifold_ox, gamma=gamma, Tc=Tc, R_gas=R_gas,
        mdot_required=mdot_required)
    result.feasible = not any(g.status == "fail" for g in result.gates)
    return result


_GRAVITY = 9.80665   # m/s^2, standard gravity for pump-head conversion


def feed_system_ledger(
    result: InjectorDesignResult,
    spec: InjectorSpec,
    *,
    Pc: float,
    regen_loss_fuel: float = 0.0,
    regen_loss_oxidizer: float = 0.0,
) -> FeedSystemLedger:
    """Per-propellant feed-pressure budget -> required pump/tank outlet pressure.

    Implements the standard liquid-rocket feed-pressure balance (Huzel & Huang,
    NASA SP-125, Ch. 1-2; Sutton & Biblarz, *Rocket Propulsion Elements*):

        P_pump_outlet >= Pc + dP_injector + dP_manifold + dP_regen(if upstream)
                          + dP_lines/valves + control_margin

    plus a pump-capacity check (mdot <= delivered capacity) and an NPSH screen
    (NPSH_available = P_tank - P_vapor >= NPSH_required; inducer sizing per NASA
    SP-8052).  Each stream is evaluated independently; only the coolant stream
    carries the regen-jacket loss.  The injector metering drop is taken from the
    sized stream (``StreamResult.dp``) so the ledger reconciles exactly with the
    upstream-pressure gate.  Pumps/tanks the user did not specify yield an
    info-level requirement rather than a pass/fail verdict.
    """
    fs_spec = spec.feed_system
    by_role = {
        "fuel": (regen_loss_fuel, fs_spec.fuel),
        "oxidizer": (regen_loss_oxidizer, fs_spec.oxidizer),
    }
    lines: dict[str, FeedLineLedger] = {}
    governing = 0.0
    for role, (regen_loss, line_spec) in by_role.items():
        stream = result.streams[role]
        feed = result.feed[role]
        rho = feed.density
        mdot = stream.mdot
        dp_inj = float(stream.dp)

        # Manifold (header + port) distribution loss.  The maldistribution
        # network's value is retained only as an INFORMATIONAL screen; the
        # amount actually charged to the pump budget is the user allowance
        # (default 0), so an unvalidated screen estimate cannot silently
        # dominate the required pump pressure.
        manifold_screen = 0.0
        if result.manifold is not None and role in result.manifold.streams:
            ml = result.manifold.streams[role].manifold_pressure_drop
            manifold_screen = float(ml) if ml == ml else 0.0
        manifold_loss = (float(line_spec.manifold_loss)
                         + float(line_spec.manifold_loss_fraction) * Pc)

        line_loss = (float(line_spec.line_loss)
                     + float(line_spec.line_loss_fraction) * Pc)
        margin = (float(line_spec.control_margin)
                  + float(line_spec.control_margin_fraction) * Pc)
        required = (Pc + dp_inj + manifold_loss + float(regen_loss)
                    + line_loss + margin)
        governing = max(governing, required)

        available = line_spec.supply_pressure
        pressure_margin = (available - required) if available is not None else None

        Q = mdot / rho if rho > 0 else float("nan")
        rise = (max(0.0, required - line_spec.tank_pressure)
                if line_spec.tank_pressure is not None else None)
        head = (rise / (rho * _GRAVITY)
                if (rise is not None and rho > 0) else None)
        power = (Q * rise / max(line_spec.pump_efficiency, 1e-6)
                 if rise is not None else None)

        cap = line_spec.flow_capacity
        cap_margin = (cap - mdot) if cap is not None else None

        npsh_avail = None
        if (line_spec.tank_pressure is not None
                and feed.vapor_pressure == feed.vapor_pressure):
            npsh_avail = line_spec.tank_pressure - feed.vapor_pressure
        npsh_req = line_spec.npsh_required
        npsh_margin = (npsh_avail - npsh_req
                       if (npsh_avail is not None and npsh_req is not None)
                       else None)

        # info when no pump data supplied; otherwise pass unless any check fails
        status = "info"
        if available is not None:
            status = "pass" if (pressure_margin is not None
                                and pressure_margin >= 0.0) else "fail"
        if cap_margin is not None and cap_margin < 0.0:
            status = "fail"
        if npsh_margin is not None and npsh_margin < 0.0:
            status = "fail"

        lines[role] = FeedLineLedger(
            role=role, chamber_pressure=Pc, injector_dp=dp_inj,
            manifold_loss=manifold_loss, manifold_screen_loss=manifold_screen,
            regen_loss=float(regen_loss),
            line_valve_loss=line_loss, control_margin=margin,
            required_outlet_pressure=required,
            available_outlet_pressure=available,
            pressure_margin=pressure_margin, density=rho,
            volumetric_flow=Q, required_pressure_rise=rise,
            required_pump_head=head, ideal_pump_power=power,
            flow_capacity=cap, capacity_margin=cap_margin,
            npsh_available=npsh_avail, npsh_required=npsh_req,
            npsh_margin=npsh_margin, status=status)

    notes = [
        "Feed-pressure balance per Huzel & Huang (NASA SP-125) and Sutton & "
        "Biblarz; NPSH screen per NASA SP-8052. Manifold loss from the "
        "two-header maldistribution solve; regen jacket loss applied to the "
        "coolant stream only.",
    ]
    return FeedSystemLedger(
        architecture=fs_spec.architecture, lines=lines,
        governing_required_pressure=governing, notes=notes)


def feed_system_gates(ledger: FeedSystemLedger) -> list[InjectorGate]:
    """Pump/feed closure gates (info-level when no pump/tank data was given)."""
    g: list[InjectorGate] = []
    for role, ln in ledger.lines.items():
        if ln.available_outlet_pressure is None:
            g.append(InjectorGate(
                f"feed_pump_pressure_{role}", "info",
                f"{role} needs >= {ln.required_outlet_pressure/1e5:.1f} bar at the "
                f"pump/tank outlet [Pc {ln.chamber_pressure/1e5:.1f} + inj "
                f"{ln.injector_dp/1e5:.2f} + manifold {ln.manifold_loss/1e5:.2f} + "
                f"regen {ln.regen_loss/1e5:.2f} + lines {ln.line_valve_loss/1e5:.2f} "
                f"+ margin {ln.control_margin/1e5:.2f} bar]; no supply pressure given "
                f"(manifold screen est. {ln.manifold_screen_loss/1e5:.2f} bar, not charged)"))
        else:
            ok = ln.pressure_margin is not None and ln.pressure_margin >= 0.0
            g.append(InjectorGate(
                f"feed_pump_pressure_{role}", "pass" if ok else "fail",
                f"{role} supply {ln.available_outlet_pressure/1e5:.1f} bar "
                f"{'>=' if ok else '<'} required "
                f"{ln.required_outlet_pressure/1e5:.1f} bar "
                f"(margin {ln.pressure_margin/1e5:+.2f} bar)"))
        if ln.capacity_margin is not None:
            ok = ln.capacity_margin >= 0.0
            g.append(InjectorGate(
                f"feed_pump_capacity_{role}", "pass" if ok else "fail",
                f"{role} pump capacity margin {ln.capacity_margin:+.4g} kg/s "
                f"({'ok' if ok else 'EXCEEDED'})"))
        if ln.npsh_margin is not None:
            g.append(InjectorGate(
                f"feed_npsh_{role}",
                "pass" if ln.npsh_margin >= 0.0 else "fail",
                f"{role} NPSH available {ln.npsh_available/1e5:.2f} bar vs required "
                f"{ln.npsh_required/1e5:.2f} bar (margin {ln.npsh_margin/1e5:+.2f} bar)"))
    return g


def evaluate_pintle_injector(
    spec: InjectorSpec,
    *,
    mdot_fuel: float,
    mdot_oxidizer: float,
    Pc: float,
    mixture_ratio: float,
    chamber_radius: float,
    chamber_length: float,
    gamma: float,
    Tc: float,
    R_gas: float,
    fuel_name: str | None = None,
    oxidizer_name: str | None = None,
    cooling: Any | None = None,
    cooling_result: dict[str, Any] | None = None,
) -> InjectorDesignResult:
    """Evaluate one pintle consistently for CLI and API/backend workflows.

    Besides calling :func:`size_pintle_injector`, this integration boundary:

    * fills missing feed identities from the thermochemistry request;
    * checks that a direct regenerative-cooling-to-injector handoff carries
      the same fuel mass flow as the engine cycle;
    * hands the calculated jacket outlet temperature/pressure to the fuel
      property resolver only when that continuity check is credible; and
    * appends the coupling gate before recomputing overall feasibility.

    A mismatched coolant flow is not silently interpreted as a bypass circuit:
    bypassed fuel would need an explicit split and mixing-temperature model.
    """
    local = copy.deepcopy(spec)
    if not local.fuel.name:
        local.fuel.name = fuel_name
    if not local.oxidizer.name:
        local.oxidizer.name = oxidizer_name

    coupling_gate = InjectorGate(
        "regen_fuel_flow_closure",
        "info",
        "no direct regenerative-cooling-to-fuel-injector handoff requested",
    )
    coupling_note = coupling_gate.detail

    cooling_method = str(getattr(cooling, "method", "none") or "none").lower()
    coolant_raw = getattr(cooling, "coolant", None)
    fuel_is_coolant = bool(
        cooling_method == "regenerative"
        and coolant_raw
        and local.fuel.name
        and canonical_coolant_name(coolant_raw)
        == canonical_coolant_name(local.fuel.name)
    )

    if fuel_is_coolant:
        coolant_mdot = float(
            getattr(cooling, "coolant_mass_flow", 0.0) or 0.0
        )
        rel_error = abs(coolant_mdot - mdot_fuel) / max(mdot_fuel, 1e-12)
        if rel_error <= _REGEN_FLOW_PASS_TOL:
            status = "pass"
        elif rel_error < _REGEN_FLOW_FAIL_TOL:
            status = "warn"
        else:
            status = "fail"
        coupling_note = (
            f"regen coolant flow {coolant_mdot:.6g} kg/s vs cycle fuel flow "
            f"{mdot_fuel:.6g} kg/s ({rel_error*100:.2f}% error; "
            f"fail >= {_REGEN_FLOW_FAIL_TOL*100:.0f}% without a bypass/mixing "
            "model)"
        )
        coupling_gate = InjectorGate(
            "regen_fuel_flow_closure", status, coupling_note
        )

        # Only a closed direct-flow path can supply an authoritative injector
        # inlet state. A warning/failure retains explicitly supplied feed
        # conditions (or the feed resolver's defaults) instead.
        if status == "pass" and cooling_result is not None:
            outlet_T = cooling_result.get("coolant_outlet_temperature")
            outlet_P = cooling_result.get("coolant_outlet_pressure")
            if local.fuel.inlet_temperature is None and outlet_T is not None:
                local.fuel.inlet_temperature = float(outlet_T)
            if local.fuel.inlet_pressure is None and outlet_P is not None:
                local.fuel.inlet_pressure = float(outlet_P)
            coupling_note += (
                f"; injector feed state uses jacket outlet "
                f"T={local.fuel.inlet_temperature:.3g} K, "
                f"P={local.fuel.inlet_pressure/1e5:.3g} bar"
                if (
                    local.fuel.inlet_temperature is not None
                    and local.fuel.inlet_pressure is not None
                )
                else "; jacket outlet state was incomplete"
            )
            coupling_gate.detail = coupling_note
        elif status == "pass":
            coupling_gate = InjectorGate(
                "regen_fuel_flow_closure",
                "info",
                coupling_note
                + "; cooling state was not evaluated, so inlet feed "
                  "conditions are retained",
            )

    result = size_pintle_injector(
        local,
        mdot_fuel=mdot_fuel,
        mdot_oxidizer=mdot_oxidizer,
        Pc=Pc,
        mixture_ratio=mixture_ratio,
        chamber_radius=chamber_radius,
        chamber_length=chamber_length,
        gamma=gamma,
        Tc=Tc,
        R_gas=R_gas,
    )
    result.gates.append(coupling_gate)
    result.notes.append(coupling_gate.detail)

    # Feed-system pressure closure (pump/tank budget + capacity + NPSH).  The
    # regen-jacket loss is charged to the fuel line only when fuel is the
    # coolant and routed to the injector through the jacket; the pump must then
    # overcome the jacket dP on top of the injector requirement.
    regen_loss_fuel = 0.0
    if fuel_is_coolant and cooling_result is not None:
        _dp_pa = cooling_result.get("coolant_pressure_drop")
        if _dp_pa is not None and _dp_pa == _dp_pa:   # not NaN
            regen_loss_fuel = float(_dp_pa)
        else:
            _dp_bar = cooling_result.get("pressure_drop_bar")
            if _dp_bar is not None and _dp_bar == _dp_bar:   # not NaN
                regen_loss_fuel = float(_dp_bar) * 1.0e5
    ledger = feed_system_ledger(
        result, local, Pc=Pc, regen_loss_fuel=regen_loss_fuel)
    result.feed_system = ledger
    result.gates.extend(feed_system_gates(ledger))
    result.notes.extend(ledger.notes)

    result.feasible = not any(g.status == "fail" for g in result.gates)
    return result


# ---------------------------------------------------------------------------
# Gates  (the required-gate list from the design brief)
# ---------------------------------------------------------------------------
def injector_gates(
    spec: InjectorSpec, r: InjectorDesignResult, *, Pc, mixture_ratio,
    dp_fuel, dp_ox, p_manifold_fuel, p_manifold_ox, gamma, Tc, R_gas,
    mdot_required,
) -> list[InjectorGate]:
    g: list[InjectorGate] = []
    mfg = spec.manufacturing
    min_feat = mfg.min_feature
    web_min = mfg.web_min if mfg.web_min is not None else min_feat
    edge_min = (mfg.edge_distance_min if mfg.edge_distance_min is not None
                else min_feat)

    # 1) mass-flow / mixture-ratio closure: delivered flow (orifice law on the
    #    final area) vs the cycle's required split.  Exact in auto sizing;
    #    reveals drift in fixed sizing.
    md_f = r.streams["fuel"].mdot
    md_o = r.streams["oxidizer"].mdot
    err_f = abs(md_f - mdot_required["fuel"]) / max(mdot_required["fuel"], 1e-12)
    err_o = abs(md_o - mdot_required["oxidizer"]) / max(
        mdot_required["oxidizer"], 1e-12)
    mr_actual = md_o / max(md_f, 1e-12)
    flow_err = max(err_f, err_o)
    # Tolerances are independent of sizing mode: a fixed geometry that misses
    # the cycle flow by more than _CLOSURE_FAIL_TOL is a genuine infeasibility,
    # not a warning (auto sizing always closes to machine precision).
    if flow_err < _CLOSURE_PASS_TOL:
        status = "pass"
    elif flow_err < _CLOSURE_FAIL_TOL:
        status = "warn"
    else:
        status = "fail"
    g.append(InjectorGate(
        "mass_flow_mixture_ratio_closure", status,
        f"delivered O/F {mr_actual:.4f} vs requested {mixture_ratio:.4f}; "
        f"flow error fuel {err_f*100:.2f}% / ox {err_o*100:.2f}% "
        f"(fail ≥ {_CLOSURE_FAIL_TOL*100:.0f}%)"))

    # 2) upstream pressure sufficiency (manifold = Pc + dp).
    for role, p_man, fs in (
        ("fuel", p_manifold_fuel, r.feed["fuel"]),
        ("oxidizer", p_manifold_ox, r.feed["oxidizer"]),
    ):
        supplied = fs.pressure
        if supplied >= p_man - 1.0:
            status, txt = "pass", (
                f"{role} feed {supplied/1e5:.1f} bar >= manifold "
                f"{p_man/1e5:.1f} bar")
        else:
            status, txt = "fail", (
                f"{role} feed {supplied/1e5:.1f} bar < manifold "
                f"{p_man/1e5:.1f} bar; requested injector delta-P cannot be "
                "produced")
        g.append(InjectorGate(f"upstream_pressure_{role}", status, txt))

    # 3) cavitation / vapor-pressure margin.
    for role, dp, p_man, fs in (
        ("fuel", dp_fuel, p_manifold_fuel, r.feed["fuel"]),
        ("oxidizer", dp_ox, p_manifold_ox, r.feed["oxidizer"]),
    ):
        if math.isnan(fs.vapor_pressure):
            g.append(InjectorGate(
                f"cavitation_{role}", "info",
                f"{role} vapor pressure unknown; cavitation not screened"))
            continue
        K = (p_man - fs.vapor_pressure) / max(dp, 1e-9)
        status = "pass" if K >= 1.5 else ("warn" if K >= 1.0 else "fail")
        g.append(InjectorGate(
            f"cavitation_{role}", status,
            f"{role} cavitation number K=(P_man-Pvap)/dp={K:.2f} "
            f"(>=1.5 desired)"))

    # 4) explicit injection state (liquid vs gas, and the gas choke state).
    for s in (r.annulus, r.slots):
        inj = s.detail.get("injection", {})
        branch = inj.get("branch", "incompressible")
        fs = r.feed[s.role]
        if branch == "incompressible":
            g.append(InjectorGate(
                f"injection_state_{s.role}", "info",
                f"{s.role} liquid ({fs.phase}); incompressible orifice law"))
        elif branch == "compressible_choked":
            g.append(InjectorGate(
                f"injection_state_{s.role}", "info",
                f"{s.role} gas ({fs.phase}) CHOKED: "
                f"Pc/P0={inj.get('pressure_ratio', float('nan')):.3f} <= "
                f"critical {inj.get('critical_pressure_ratio', float('nan')):.3f}; "
                f"sonic injection, flow set by upstream P0/T0"))
        else:
            g.append(InjectorGate(
                f"injection_state_{s.role}", "info",
                f"{s.role} gas ({fs.phase}) subsonic: "
                f"Pc/P0={inj.get('pressure_ratio', float('nan')):.3f} > critical "
                f"{inj.get('critical_pressure_ratio', float('nan')):.3f}, "
                f"M={inj.get('exit_mach', float('nan')):.2f} "
                f"(weak feed-system decoupling near critical)"))

    # 5) Cd / L-over-D / hydraulic-flip / correlation domain.
    for s in (r.annulus, r.slots):
        lod = s.detail.get("length_over_dh", float("nan"))
        if math.isnan(lod):
            status, txt = "info", (
                f"{s.role} {s.geometry}: L/D unknown (supply slot/body "
                f"depth); Cd={s.cd:.2f}")
        elif lod < 1.0:
            status, txt = "warn", (
                f"{s.role} {s.geometry}: L/D={lod:.1f} < 1 (sharp-edged, "
                f"hydraulic-flip prone); Cd={s.cd:.2f}")
        else:
            status, txt = "pass", (
                f"{s.role} {s.geometry}: L/D={lod:.1f}, Cd={s.cd:.2f}")
        g.append(InjectorGate(f"orifice_domain_{s.geometry}", status, txt))

    # 6) minimum slot width / web / annulus gap / edge distance.
    w = r.slots.detail["slot_width"]
    gap = r.annulus.detail["gap"]
    g.append(InjectorGate(
        "min_slot_width", "pass" if w >= min_feat else "fail",
        f"slot width {w*1e3:.3f} mm vs floor {min_feat*1e3:.3f} mm"))
    g.append(InjectorGate(
        "min_web", "pass" if r.minimum_web >= web_min else "fail",
        f"ligament/web {r.minimum_web*1e3:.3f} mm vs floor {web_min*1e3:.3f} mm"))
    g.append(InjectorGate(
        "min_annulus_gap", "pass" if gap >= min_feat else "fail",
        f"annulus gap {gap*1e3:.3f} mm vs floor {min_feat*1e3:.3f} mm"))
    g.append(InjectorGate(
        "slot_blockage", "pass" if r.blockage_factor < 1.0 else "fail",
        f"slot blockage N*w/(pi*Dp)={r.blockage_factor:.2f} (<1 required)"))

    # 7) annulus concentricity / tolerance sensitivity.
    ecc = mfg.concentricity_tolerance / max(gap, 1e-12)
    status = "pass" if ecc < 0.10 else ("warn" if ecc < 0.25 else "fail")
    g.append(InjectorGate(
        "annulus_concentricity", status,
        f"eccentricity tol/gap={ecc:.2f} (gap {gap*1e3:.3f} mm, tol "
        f"{mfg.concentricity_tolerance*1e3:.3f} mm); flow maldistribution "
        f"~2x eccentricity"))

    # 8) manifold maldistribution — one gate per propellant manifold.
    if r.manifold is not None:
        for role, m in r.manifold.streams.items():
            spread = m.maldistribution_fraction
            if not (spread == spread):  # NaN -> unscreened
                g.append(InjectorGate(
                    f"manifold_maldistribution_{role}", "info",
                    f"{role} manifold ({m.feeds}) not screened ({m.status})"))
                continue
            status = ("pass" if spread < 0.10 else
                      "warn" if spread < 0.25 else "fail")
            g.append(InjectorGate(
                f"manifold_maldistribution_{role}", status,
                f"{role} manifold ({m.feeds}, {m.port_count} ports) "
                f"element flow spread {spread*100:.1f}% "
                f"[{m.min_flow_ratio:.2f}–{m.max_flow_ratio:.2f}×]"))

    # 9) spray-wall clearance.
    if not math.isfinite(r.spray_wall_axial_distance):
        g.append(InjectorGate(
            "spray_wall_clearance", "warn",
            "spray nearly axial; no wall interception within the chamber"))
    else:
        frac = r.spray_wall_axial_distance / max(r.chamber_length, 1e-9)
        if r.spray_half_angle_deg >= 90.0:
            status, txt = "fail", (
                f"spray half-angle {r.spray_half_angle_deg:.0f} deg >= 90 "
                f"(reversed toward the face)")
        elif frac < 0.05:
            status, txt = "warn", (
                f"spray hits wall {r.spray_wall_axial_distance*1e3:.1f} mm "
                f"from tip ({frac*100:.0f}% of Lc); local gouging risk "
                f"(cf. Apollo oxidizer fans)")
        else:
            status, txt = "pass", (
                f"spray half-angle {r.spray_half_angle_deg:.0f} deg meets wall "
                f"at {r.spray_wall_axial_distance*1e3:.1f} mm ({frac*100:.0f}% "
                f"of Lc)")
        g.append(InjectorGate("spray_wall_clearance", status, txt))

    # 9a) combustion-development length vs available chamber length (the L*/
    #     injector-quality coupling; vaporization-limited c* surrogate).
    if r.atomization is not None:
        at = r.atomization
        # A vaporization-limited SURROGATE: warn (iterate L*/geometry/Δp) but
        # never hard-fail export over a screening estimate (SP-8089).
        m = at.development_margin
        if not math.isfinite(m):
            status, txt = "info", "combustion length indeterminate"
        else:
            status = "pass" if m >= 1.0 else "warn"
            hint = ("" if m >= 1.0 else
                    "; increase L*/Δp/velocity or change stream assignment")
            smd_um = at.streams[at.limiting_role].sauter_mean_diameter * 1e6
            txt = (f"{at.limiting_role} combustion-development length "
                   f"{at.combustion_length*1e3:.0f} mm vs chamber "
                   f"{at.available_chamber_length*1e3:.0f} mm (margin {m:.2f}{hint}); "
                   f"SMD≈{smd_um:.0f} µm, predicted η_c*≈"
                   f"{at.predicted_cstar_efficiency:.2f} "
                   f"(vaporization-limited surrogate)")
        g.append(InjectorGate("combustion_development_length", status, txt))

    # 9b) target momentum ratio (when requested): compare achieved TMR.
    if spec.target_momentum_ratio is not None:
        tgt = spec.target_momentum_ratio
        rel = abs(r.total_momentum_ratio - tgt) / max(abs(tgt), 1e-9)
        status = "pass" if rel < 0.15 else ("warn" if rel < 0.40 else "fail")
        g.append(InjectorGate(
            "target_momentum_ratio", status,
            f"achieved TMR {r.total_momentum_ratio:.2f} vs target {tgt:.2f} "
            f"({rel*100:.0f}% off); active targeting needs dp-split/geometry "
            f"adjustment (throttle solver)"))

    # 9c) pintle-tip radius sanity (cannot exceed the pintle radius).
    if spec.geometry.tip_radius is not None:
        rmax = 0.5 * r.pintle_diameter
        g.append(InjectorGate(
            "pintle_tip_radius",
            "pass" if spec.geometry.tip_radius <= rmax else "fail",
            f"tip radius {spec.geometry.tip_radius*1e3:.2f} mm vs pintle radius "
            f"{rmax*1e3:.2f} mm"))

    # 9d) impingement distance (openings -> stream interaction).
    if spec.geometry.impingement_distance is not None:
        imp = spec.geometry.impingement_distance
        g.append(InjectorGate(
            "impingement_distance",
            "pass" if 0.0 < imp < r.chamber_length else "warn",
            f"impingement {imp*1e3:.1f} mm from openings to interaction; "
            f"spray origin offset folded into wall clearance"))

    # 9e) injector face OD must cover the chamber bore; edge-distance land.
    Do = r.annulus.detail["outer_diameter"]
    if spec.geometry.face_od is not None:
        chamber_d = 2.0 * r.chamber_radius
        g.append(InjectorGate(
            "injector_face_od",
            "pass" if spec.geometry.face_od >= chamber_d else "warn",
            f"face OD {spec.geometry.face_od*1e3:.1f} mm vs chamber bore "
            f"{chamber_d*1e3:.1f} mm"))
        land = 0.5 * spec.geometry.face_od - 0.5 * Do
        g.append(InjectorGate(
            "edge_distance", "pass" if land >= edge_min else "fail",
            f"annulus-to-face land {land*1e3:.2f} mm vs edge floor "
            f"{edge_min*1e3:.2f} mm"))
    else:
        g.append(InjectorGate(
            "edge_distance", "info",
            "edge distance not checked (supply face OD via --injector-face-od)"))

    # 10) face / pintle-tip thermal margin (recirculation + series circuit).
    if r.thermal is not None:
        t = r.thermal
        mg = t.governing_margin
        # Screening recirculation estimate -> warn-capped (never hard-fail on
        # a rough chamber-side heat balance); report the limiting surface.
        status = "pass" if mg >= 1.2 else "warn"
        g.append(InjectorGate(
            "face_tip_thermal_margin", status,
            f"{t.limiting} governs: T_wg≈"
            f"{(t.tip_wall_temperature if t.limiting=='tip' else t.face_wall_temperature):.0f} K "
            f"vs limit {t.wall_temperature_limit:.0f} K (margin {mg:.2f}); "
            f"recirculation+series-circuit screen, needs CFD"))

    # 11) throttle-point mixture-ratio drift (fixed geometry).
    g.append(InjectorGate(
        "throttle_mr_drift", "info",
        "fixed areas: dp ~ mdot^2, so at 50% flow dp -> ~25%; O/F is "
        "preserved only while both dp-fractions scale together — a movable "
        "pintle is needed for deep throttling"))

    # 12) stability: chug (feed decoupling) + chamber acoustics + n-τ band.
    if r.stability is not None:
        st = r.stability
        # chug gate (a real, well-grounded feed-decoupling rule)
        chug_status = ("pass" if st.injector_decoupling_fraction >= 0.20 else
                       "warn" if st.injector_decoupling_fraction >= 0.10 else
                       "fail")
        g.append(InjectorGate(
            "feed_system_chug", chug_status,
            f"injector decoupling min(χ)={st.injector_decoupling_fraction:.2f} "
            f"-> {st.chug_status}"))
        # acoustic modes (informational)
        g.append(InjectorGate(
            "chamber_acoustic_modes", "info",
            f"a={st.sound_speed:.0f} m/s -> L1 {st.f_L1:.0f} Hz, "
            f"L2 {st.f_L2:.0f} Hz, T1 {st.f_T1:.0f} Hz, R1 {st.f_R1:.0f} Hz"))
        # n-τ reduced-frequency band (screening)
        nt_status = "warn" if st.sensitive_band else "info"
        g.append(InjectorGate(
            "ntau_coupling", nt_status,
            f"combustion lag τ≈{st.combustion_time_lag*1e3:.2f} ms, "
            f"τ·f_L1={st.reduced_frequency_L1:.2f}"
            + (" (in the sensitive n-τ band — high-frequency risk)"
               if st.sensitive_band else " (outside the first sensitive band)")
            + "; screening, needs the combustion response function"))

    # 13) mandatory cold-flow + hot-fire validation status.
    g.append(InjectorGate(
        "validation_status", "warn",
        "cold-flow and hot-fire validation REQUIRED and not performed "
        "(NASA SP-8089: pintle spray distributions need cold-flow testing)"))

    return g
