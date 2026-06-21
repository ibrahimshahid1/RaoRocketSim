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

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


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
class InjectorGate:
    name: str
    status: str                 # "pass" | "warn" | "fail" | "info"
    detail: str

    @property
    def ok(self) -> bool:
        return self.status in ("pass", "info")


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


def _validate_injector_spec(spec, mdot_fuel, mdot_oxidizer, Pc, mixture_ratio):
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
    if spec.manufacturing.min_feature <= 0.0:
        errs.append(
            f"min_feature must be > 0, got {spec.manufacturing.min_feature}")
    if spec.sizing == "fixed":
        if geo.annulus_gap is None or geo.slot_width is None:
            errs.append(
                "fixed sizing requires both annulus_gap and slot_width "
                "(otherwise it would silently auto-size the geometry)")
        if geo.annulus_gap is not None and geo.annulus_gap <= 0.0:
            errs.append("annulus_gap must be > 0")
        if geo.slot_width is not None and geo.slot_width <= 0.0:
            errs.append("slot_width must be > 0")
    if mdot_fuel <= 0.0 or mdot_oxidizer <= 0.0:
        errs.append("fuel and oxidizer mass flows must be positive")
    if Pc <= 0.0:
        errs.append("chamber pressure must be positive")
    if mixture_ratio <= 0.0:
        errs.append("mixture_ratio must be positive")
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
        Tcrit=678.0, Pcrit=2.2e6,
        source="RP-1/Jet-A class (Sutton & Biblarz RPE; NASA SP-8087): "
        "rho~810 kg/m^3, mu~1.6e-3 Pa.s, sigma~0.023 N/m, Pvap~2 kPa @298 K "
        "(constant-property screening)",
    ),
    "mmh": dict(
        rho=874.0, mu=0.775e-3, sigma=0.0341, Pvap=6.6e3, T_ref=298.0,
        Tcrit=585.0, Pcrit=8.2e6,
        source="MMH (Sutton & Biblarz RPE; CRC): rho 874, mu 0.78e-3, "
        "sigma 0.034, Pvap 6.6 kPa @298 K (constant-property screening)",
    ),
    "n2o4": dict(
        rho=1443.0, mu=0.42e-3, sigma=0.0267, Pvap=96.0e3, T_ref=293.0,
        Tcrit=431.0, Pcrit=10.1e6,
        source="N2O4/NTO (Sutton & Biblarz RPE): rho 1443, mu 0.42e-3, "
        "sigma 0.0267, Pvap 96 kPa @293 K (volatile; low ambient cavitation "
        "margin; constant-property screening)",
    ),
    "udmh": dict(
        rho=791.0, mu=0.492e-3, sigma=0.0289, Pvap=16.3e3, T_ref=298.0,
        Tcrit=523.0, Pcrit=5.4e6,
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
        liquid_ok, phase, reason = _classify_phase(
            T, P, Pvap, Tcrit, spec.phase, subcool_margin
        )
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
            reason=reason,
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
    # Saturation properties (only meaningful below the critical point).
    if T < Tcrit:
        try:
            Pvap = float(PropsSI("P", "T", T, "Q", 0, fluid))
            sigma = float(PropsSI("SURFACE_TENSION", "T", T, "Q", 0, fluid))
        except Exception:
            Pvap, sigma = float("nan"), float("nan")
    else:
        Pvap, sigma = float("nan"), float("nan")
    liquid_ok, phase, reason = _classify_phase(
        T, P, Pvap, Tcrit, spec.phase, subcool_margin
    )
    # Override surface tension if supplied (needed when supercritical etc.).
    if spec.surface_tension is not None:
        sigma = float(spec.surface_tension)
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
        liquid_ok=liquid_ok, reason=reason,
    )


def _classify_phase(T, P, Pvap, Tcrit, requested, subcool_margin):
    """Return (liquid_ok, phase, reason) for the liquid/liquid MVP."""
    if Tcrit is not None and T >= 0.98 * Tcrit:
        phase = "supercritical"
        return False, phase, (
            f"T={T:.1f} K is at/above 0.98*Tcrit={0.98*Tcrit:.1f} K "
            f"(supercritical/near-critical); needs a real-fluid branch"
        )
    if not math.isnan(Pvap):
        if P <= Pvap:
            return False, "gas", (
                f"feed pressure {P/1e5:.2f} bar <= vapor pressure "
                f"{Pvap/1e5:.2f} bar; the stream is gaseous/flashing"
            )
        if P < (1.0 + subcool_margin) * Pvap:
            return False, "liquid", (
                f"feed pressure {P/1e5:.2f} bar within {subcool_margin*100:.0f}% "
                f"of vapor pressure {Pvap/1e5:.2f} bar (cavitation/flashing risk)"
            )
    if requested == "gas":
        return False, "gas", "phase forced to gas (unsupported in liquid MVP)"
    return True, "liquid", ""


# ---------------------------------------------------------------------------
# Hydraulic sizing
# ---------------------------------------------------------------------------
def _orifice_area(mdot: float, cd: float, rho: float, dp: float) -> float:
    """Geometric flow area from the incompressible orifice law."""
    if dp <= 0.0:
        raise ValueError("injector pressure drop must be positive")
    return mdot / (cd * math.sqrt(2.0 * rho * dp))


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


def _stream_numbers(role, geom, mdot, dp, cd, area, dh, rho, mu, sigma):
    v = mdot / (rho * area)
    re = rho * v * dh / mu
    we = rho * v * v * dh / sigma if sigma > 0 else float("nan")
    oh = mu / math.sqrt(rho * sigma * dh) if sigma > 0 else float("nan")
    return StreamResult(
        role=role, geometry=geom, mdot=mdot, dp=dp, cd=cd, area=area,
        velocity=v, hydraulic_diameter=dh, reynolds=re, weber=we, ohnesorge=oh,
    )


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
    _validate_injector_spec(spec, mdot_fuel, mdot_oxidizer, Pc, mixture_ratio)
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

    # Liquid-only guard (MVP). Reject before sizing so the message is clean.
    for role, fs in feed.items():
        if not fs.liquid_ok:
            raise InjectorUnsupportedState(
                f"{role} feed ('{fs.name}') is not a usable liquid: {fs.reason}. "
                f"The liquid/liquid MVP does not size gas/near-critical "
                f"injection (compressible branch deferred)."
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
    if fixed and spec.geometry.annulus_gap is not None:
        gap = spec.geometry.annulus_gap
        Do = Dp + 2.0 * gap
        A_a = math.pi / 4.0 * (Do * Do - Dp * Dp)
        ann_geom = {"inner_diameter": Dp, "outer_diameter": Do, "gap": gap,
                    "hydraulic_diameter": Do - Dp}
    else:
        A_a = _orifice_area(m_a_req, cd_a, fs_a.density, dp_a)
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
    m_a = cd_a * A_a * math.sqrt(2.0 * fs_a.density * dp_a)  # delivered
    annulus = _stream_numbers(
        axial, "annulus", m_a, dp_a, cd_a, A_a, ann_geom["hydraulic_diameter"],
        fs_a.density, fs_a.viscosity, fs_a.surface_tension)
    annulus.detail = ann_geom

    # ----- radial slot stream -----
    m_r_req, dp_r, cd_r, fs_r = streams_in[radial]
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
        A_r = _orifice_area(m_r_req, cd_r, fs_r.density, dp_r)
        slot_geom = _slots_from_area(
            A_r, spec.geometry.slot_count, spec.geometry.slot_aspect_ratio,
            Dp, spec.geometry.slot_depth, spec.geometry.slot_length_over_dh)
        slot_geom["area"] = A_r
    m_r = cd_r * A_r * math.sqrt(2.0 * fs_r.density * dp_r)  # delivered
    slots = _stream_numbers(
        radial, "slots", m_r, dp_r, cd_r, A_r, slot_geom["hydraulic_diameter"],
        fs_r.density, fs_r.viscosity, fs_r.surface_tension)
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
    result.gates = injector_gates(
        spec, result, Pc=Pc, mixture_ratio=mixture_ratio,
        dp_fuel=dp_fuel, dp_ox=dp_ox, p_manifold_fuel=p_manifold_fuel,
        p_manifold_ox=p_manifold_ox, gamma=gamma, Tc=Tc, R_gas=R_gas,
        mdot_required=mdot_required)
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
            status, txt = "warn", (
                f"{role} feed {supplied/1e5:.1f} bar < manifold "
                f"{p_man/1e5:.1f} bar (add feed/regen head)")
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

    # 4) explicit liquid / choke state.
    g.append(InjectorGate(
        "injection_state", "info",
        "both streams liquid (incompressible orifice law); gas/choked "
        "injection not in this MVP"))

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

    # 8) manifold maldistribution (annular header network, radial stream).
    g.append(_manifold_maldistribution_gate(r, dp_ox if r.radial_stream ==
                                            "oxidizer" else dp_fuel))

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

    # 10) face / pintle-tip thermal margin (screening indicator).
    g.append(InjectorGate(
        "face_tip_thermal_margin", "info",
        "face/tip heat load is regen/film dependent; pintle-tip and face "
        "cooling not yet coupled (screening indicator only)"))

    # 11) throttle-point mixture-ratio drift (fixed geometry).
    g.append(InjectorGate(
        "throttle_mr_drift", "info",
        "fixed areas: dp ~ mdot^2, so at 50% flow dp -> ~25%; O/F is "
        "preserved only while both dp-fractions scale together — a movable "
        "pintle is needed for deep throttling"))

    # 12) feed/chamber acoustic-frequency screening.
    a_gas = math.sqrt(gamma * R_gas * Tc)
    f_L1 = a_gas / (2.0 * max(r.chamber_length, 1e-9))
    f_T1 = 1.8412 * a_gas / (math.pi * 2.0 * r.chamber_radius)
    g.append(InjectorGate(
        "acoustic_screen", "info",
        f"chamber a={a_gas:.0f} m/s -> f_L1~{f_L1:.0f} Hz, f_T1~{f_T1:.0f} Hz; "
        f"separate feed-coupling/admittance screening still required"))

    # 13) mandatory cold-flow + hot-fire validation status.
    g.append(InjectorGate(
        "validation_status", "warn",
        "cold-flow and hot-fire validation REQUIRED and not performed "
        "(NASA SP-8089: pintle spray distributions need cold-flow testing)"))

    return g


def _manifold_maldistribution_gate(r: InjectorDesignResult, dp) -> InjectorGate:
    """Run the annular-header network on the slotted stream as a screen."""
    try:
        from raosim.thermofluids import solve_annular_manifold_network
        net = solve_annular_manifold_network(
            channel_count=max(int(r.slot_count), 2),
            ports_per_manifold=4,
            total_mass_flow=r.slots.mdot,
            density=r.feed[r.radial_stream].density,
            channel_pressure_drop=max(dp, 1.0),
            channel_total_area=r.slots.area,
            manifold_radius=0.5 * r.pintle_diameter,
        )
        spread = net["maldistribution_fraction"]
        status = "pass" if spread < 0.10 else ("warn" if spread < 0.25
                                               else "fail")
        return InjectorGate(
            "manifold_maldistribution", status,
            f"slot-to-slot flow spread {spread*100:.1f}% across the annular "
            f"header ({net['status']})")
    except Exception as exc:
        return InjectorGate(
            "manifold_maldistribution", "info",
            f"manifold network not screened ({type(exc).__name__})")
