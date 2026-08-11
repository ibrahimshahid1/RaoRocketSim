"""One host-side definition of every input and convention both pipelines use.

``ResolvedEngineInputs`` is remediation item 9.  It is deliberately written to
be *contained by* a future ``ResolvedEngineJob`` rather than to sit beside one:
a job adds workflow intent (analyze / optimize / pareto), variable policies,
targets, and an output policy on top of these physical inputs.  Building the
two as peers would create a third competing definition of the same
conventions, which is the failure this contract exists to end.

The invariant is that neither snapshot adapter may reconstruct an input from a
default.  Every field here is resolved once, on the host, in SI, and both the
differentiable core and the traditional pipeline read the same object.  A
missing value is therefore an explicit ``None`` with a stated reason, never a
silent fallback that lets the two paths diverge.

Scope note: this is the *input* contract.  ``EngineState`` v2 remains the
numerical state and ``EngineAnalysisSnapshot`` v2 remains the output contract.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

__all__ = [
    "RESOLVED_INPUTS_SCHEMA_VERSION",
    "InjectorInputs",
    "MaterialInputs",
    "PerformanceConventions",
    "PropellantInputs",
    "ResolvedEngineInputs",
    "ContourInputs",
    "FeedInputs",
    "ThermalInputs",
    "crosscheck_design_input",
    "resolve_engine_inputs",
]

#: Bumped whenever the meaning of a field changes.  A stored resolved-input
#: record from an older version must be re-resolved, never relabelled.
RESOLVED_INPUTS_SCHEMA_VERSION = "1.0.0"


def _finite(name: str, value: Any) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return out


def _positive(name: str, value: Any) -> float:
    out = _finite(name, value)
    if out <= 0.0:
        raise ValueError(f"{name} must be positive; got {out!r}")
    return out


@dataclass(frozen=True)
class PropellantInputs:
    """Identities and the effective mixture ratio actually being analyzed.

    ``mixture_ratio`` is the *effective* O/F after the design-layout resolver,
    so a variable-O/F optimum lands here rather than the mission nominal.
    """

    combination: str
    oxidizer: str
    fuel: str
    coolant: str
    mixture_ratio: float
    mixture_ratio_source: str          # "mission_nominal" | "optimized" | "pinned"
    fuel_density: float
    oxidizer_density: float

    def __post_init__(self) -> None:
        _positive("mixture_ratio", self.mixture_ratio)
        _positive("fuel_density", self.fuel_density)
        _positive("oxidizer_density", self.oxidizer_density)


@dataclass(frozen=True)
class PerformanceConventions:
    """The split-efficiency convention, stated once for both pipelines.

    ``eta_cstar_effective`` already carries any film penalty, so a consumer
    must not re-apply one.  ``ideal_basis`` names which ideal quantity the
    delivered values are referred to, because comparing a delivered Isp
    against a differently-based ideal is the classic silent parity error.
    """

    eta_cstar_nominal: float
    eta_cstar_effective: float
    eta_CF: float
    ideal_basis: str = "frozen_chamber_ideal"
    delivered_convention: str = "eta_cstar_times_eta_CF"

    def __post_init__(self) -> None:
        _positive("eta_cstar_nominal", self.eta_cstar_nominal)
        _positive("eta_cstar_effective", self.eta_cstar_effective)
        _positive("eta_CF", self.eta_CF)

    @property
    def eta_Isp(self) -> float:
        return self.eta_cstar_effective * self.eta_CF


@dataclass(frozen=True)
class ContourInputs:
    """Geometry conventions.  ``Ru/Rt`` and ``Rd/Rt`` are kept separate."""

    epsilon: float
    length_pct: float
    contraction_ratio: float
    l_star: float
    throat_upstream_radius_ratio: float
    throat_downstream_radius_ratio: float
    convergent_half_angle_deg: float
    method: str
    provider_id: str = "rao_top_bezier_fixed_topology"

    def __post_init__(self) -> None:
        _positive("epsilon", self.epsilon)
        _positive("contraction_ratio", self.contraction_ratio)
        _positive("l_star", self.l_star)


@dataclass(frozen=True)
class MaterialInputs:
    """The liner/closeout selection, resolved atomically upstream.

    ``*_selection_resolved`` is False when the class-default wall constants
    are in force.  Those defaults are not any one catalog alloy, so a report
    must not name one.
    """

    liner_name: str | None
    closeout_name: str | None
    liner_conductivity: float
    liner_density: float
    liner_elastic_modulus: float
    liner_thermal_expansion: float
    liner_poisson_ratio: float
    liner_allowable_stress: float
    liner_structural_fos: float
    liner_max_gas_side_wall_temp: float
    closeout_density: float
    closeout_yield_strength: float
    closeout_elastic_modulus: float
    closeout_poisson_ratio: float
    closeout_structural_fos: float

    @property
    def liner_selection_resolved(self) -> bool:
        return self.liner_name is not None

    @property
    def closeout_selection_resolved(self) -> bool:
        return self.closeout_name is not None

    @property
    def liner_yield_strength(self) -> float:
        """Recover catalog yield without re-applying the factor of safety."""
        return self.liner_allowable_stress * self.liner_structural_fos


@dataclass(frozen=True)
class ThermalInputs:
    """Regenerative/film split and the coolant-side conventions."""

    channel_count: int
    channel_width: float
    channel_height: float
    channel_roughness: float
    wall_thickness: float
    regen_mass_flow: float
    film_mass_flow: float
    coolant_inlet_temperature: float
    coolant_wall_temperature_limit: float
    coolant_density: float
    coolant_viscosity: float
    coolant_conductivity: float
    coolant_cp: float
    coolant_property_backend: str = "constant"

    def __post_init__(self) -> None:
        _positive("channel_width", self.channel_width)
        _positive("channel_height", self.channel_height)
        _positive("wall_thickness", self.wall_thickness)
        if self.regen_mass_flow < 0.0 or self.film_mass_flow < 0.0:
            raise ValueError("regen and film mass flows must be non-negative")


@dataclass(frozen=True)
class InjectorInputs:
    """Both pressure-drop fractions and both discharge coefficients."""

    kind: str
    architecture: str
    fuel_dp_fraction: float
    oxidizer_dp_fraction: float
    fuel_cd: float
    oxidizer_cd: float
    pintle_diameter: float
    slot_count: int
    slot_aspect_ratio: float
    deflector_angle_deg: float
    radial_stream: str = "fuel"

    def __post_init__(self) -> None:
        _positive("fuel_dp_fraction", self.fuel_dp_fraction)
        _positive("oxidizer_dp_fraction", self.oxidizer_dp_fraction)
        _positive("fuel_cd", self.fuel_cd)
        _positive("oxidizer_cd", self.oxidizer_cd)


@dataclass(frozen=True)
class FeedInputs:
    """Tank pressures, line losses and the pump assumptions."""

    architecture: str
    fuel_tank_pressure: float
    oxidizer_tank_pressure: float
    line_pressure_loss: float
    pump_efficiency: float
    pump_speed_rpm: float

    def __post_init__(self) -> None:
        _positive("fuel_tank_pressure", self.fuel_tank_pressure)
        _positive("oxidizer_tank_pressure", self.oxidizer_tank_pressure)
        _positive("pump_efficiency", self.pump_efficiency)


@dataclass(frozen=True)
class ResolvedEngineInputs:
    """The frozen contract both pipelines consume.

    ``model_identities`` carries the property-table content hash and the
    model/provider ids, so a parity comparison can prove the two paths used
    the same thermochemistry rather than merely the same numbers.
    """

    schema_version: str
    target_thrust: float
    chamber_pressure: float
    ambient_pressure: float
    burn_time: float
    total_mass_flow: float
    propellant: PropellantInputs
    performance: PerformanceConventions
    contour: ContourInputs
    material: MaterialInputs
    thermal: ThermalInputs
    injector: InjectorInputs
    feed: FeedInputs
    model_identities: Mapping[str, Any] = field(default_factory=dict)
    unavailable: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _positive("target_thrust", self.target_thrust)
        _positive("chamber_pressure", self.chamber_pressure)
        _finite("ambient_pressure", self.ambient_pressure)
        _positive("burn_time", self.burn_time)
        _positive("total_mass_flow", self.total_mass_flow)

    # -- derived splits, defined once so neither pipeline re-derives them --- #
    @property
    def fuel_mass_flow(self) -> float:
        return self.total_mass_flow / (1.0 + self.propellant.mixture_ratio)

    @property
    def oxidizer_mass_flow(self) -> float:
        return self.total_mass_flow - self.fuel_mass_flow

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        """Content hash of the resolved inputs, for replay and parity proof."""
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def assert_flow_closure(self, *, rel_tol: float = 1.0e-9) -> None:
        """The regen and film branches must close to total fuel flow."""
        branch = self.thermal.regen_mass_flow + self.thermal.film_mass_flow
        if not math.isclose(branch, self.fuel_mass_flow, rel_tol=rel_tol,
                            abs_tol=1.0e-12):
            raise ValueError(
                "regenerative jacket flow + film flow must equal total fuel "
                f"flow; got {branch!r} against {self.fuel_mass_flow!r}. A "
                "cooling fraction below 1 needs an explicit third fuel-bypass "
                "branch before parity post-processing."
            )


def resolve_engine_inputs(
    design: Mapping[str, Any],
    mission: Any,
    *,
    effective_of: float,
    of_source: str,
    total_mass_flow: float,
    eta_cstar_effective: float | None = None,
    contour_method: str = "bezier",
    surfaces: Any = None,
) -> ResolvedEngineInputs:
    """Resolve one host-side input contract from a solved MDO design point.

    ``effective_of`` must already be the authoritative value from the design
    layout resolver -- this function does not re-derive mixture ratio, because
    re-deriving it in a second place is exactly how the optimized value was
    lost before.
    """

    from raosim.mdo.propellants import get_propellant

    d = dict(design)
    of = _positive("effective_of", effective_of)
    combination = str(mission.propellant_name)
    try:
        spec = get_propellant(combination)
        oxidizer, fuel = spec.name.split("/", 1)
        coolant = str(spec.coolant_name)
    except (KeyError, ValueError):
        # A custom OXIDIZER/FUEL pair supplied with its own validated table is
        # legitimate; an unresolvable coolant is recorded, never guessed.
        parts = combination.split("/", 1)
        oxidizer, fuel = (parts + ["unspecified"])[:2]
        coolant = ""

    film_frac = float(d.get("film_frac", 0.0))
    if eta_cstar_effective is None:
        eta_cstar_effective = float(mission.eta_cstar) * (
            1.0 - float(mission.film_cstar_penalty) * film_frac
        )
    mdot_total = _positive("total_mass_flow", total_mass_flow)
    mdot_fuel = mdot_total / (1.0 + of)
    mdot_film = mdot_fuel * film_frac
    mdot_regen = float(mission.cooling_fraction) * mdot_fuel * (1.0 - film_frac)

    unavailable: dict[str, str] = {}
    if not coolant:
        unavailable["coolant"] = (
            f"propellant {combination!r} is not in the catalog, so no coolant "
            "identity resolves"
        )

    rho_closeout = getattr(mission, "rho_closeout", None)
    material = MaterialInputs(
        liner_name=getattr(mission, "liner_material_name", None),
        closeout_name=getattr(mission, "closeout_material_name", None),
        liner_conductivity=float(mission.k_wall),
        liner_density=float(mission.rho_wall),
        liner_elastic_modulus=float(mission.liner_E),
        liner_thermal_expansion=float(mission.liner_alpha),
        liner_poisson_ratio=float(mission.liner_poisson),
        liner_allowable_stress=float(mission.liner_sigma_allow),
        liner_structural_fos=float(mission.liner_structural_fos),
        liner_max_gas_side_wall_temp=float(mission.liner_T_wg_max),
        closeout_density=float(
            rho_closeout if rho_closeout is not None else mission.rho_wall
        ),
        closeout_yield_strength=float(mission.closeout_sigma_yield),
        closeout_elastic_modulus=float(mission.closeout_E),
        closeout_poisson_ratio=float(mission.closeout_poisson),
        closeout_structural_fos=float(mission.closeout_structural_fos),
    )
    if material.liner_name is None:
        unavailable["liner_material"] = (
            "no catalog record backs the traced wall constants; they are "
            "class defaults and correspond to no single alloy"
        )

    identities: dict[str, Any] = {
        "resolved_inputs_schema": RESOLVED_INPUTS_SCHEMA_VERSION,
        "contour_provider": "rao_top_bezier_fixed_topology",
        "thermochemistry": (
            "mdo_cea_surface" if mission.cea_table_path
            else "mdo_constant_property"
        ),
        "coolant_property_backend": "constant",
    }
    if surfaces is not None:
        identities["property_table_sha256"] = str(
            getattr(surfaces, "content_sha256", "") or ""
        )
        identities["property_domain_policy"] = str(
            getattr(surfaces, "domain_policy", "") or ""
        )

    return ResolvedEngineInputs(
        schema_version=RESOLVED_INPUTS_SCHEMA_VERSION,
        target_thrust=float(mission.thrust),
        chamber_pressure=float(d["Pc"]),
        ambient_pressure=float(mission.Pa),
        burn_time=float(mission.burn_time),
        total_mass_flow=mdot_total,
        propellant=PropellantInputs(
            combination=combination,
            oxidizer=oxidizer,
            fuel=fuel,
            coolant=coolant,
            mixture_ratio=of,
            mixture_ratio_source=str(of_source),
            fuel_density=float(mission.rho_fuel),
            oxidizer_density=float(mission.rho_ox),
        ),
        performance=PerformanceConventions(
            eta_cstar_nominal=float(mission.eta_cstar),
            eta_cstar_effective=float(eta_cstar_effective),
            eta_CF=float(mission.eta_CF),
        ),
        contour=ContourInputs(
            epsilon=float(d["eps"]),
            length_pct=float(mission.length_pct),
            contraction_ratio=float(mission.contraction_ratio),
            l_star=float(mission.l_star),
            throat_upstream_radius_ratio=float(
                getattr(mission, "throat_ru_factor", 1.5)
            ),
            throat_downstream_radius_ratio=float(
                getattr(mission, "throat_rd_factor", 0.382)
            ),
            convergent_half_angle_deg=float(mission.converging_half_angle_deg),
            method=str(contour_method),
        ),
        material=material,
        thermal=ThermalInputs(
            channel_count=int(mission.n_channels),
            channel_width=float(d["channel_width"]),
            channel_height=float(d["channel_height"]),
            channel_roughness=float(mission.channel_roughness),
            wall_thickness=float(d["t_wall"]),
            regen_mass_flow=mdot_regen,
            film_mass_flow=mdot_film,
            coolant_inlet_temperature=float(mission.coolant_temperature),
            coolant_wall_temperature_limit=float(
                mission.rp1_coking_wall_temp_K
            ),
            coolant_density=float(mission.rho_cool),
            coolant_viscosity=float(mission.mu_cool),
            coolant_conductivity=float(mission.k_cool),
            coolant_cp=float(mission.cp_cool),
        ),
        injector=InjectorInputs(
            kind="pintle",
            architecture="fixed_discrete",
            fuel_dp_fraction=float(d["dp_f_frac"]),
            oxidizer_dp_fraction=float(d["dp_o_frac"]),
            fuel_cd=float(mission.injector_cd_fuel),
            oxidizer_cd=float(mission.injector_cd_ox),
            pintle_diameter=float(d["D_pintle"]),
            slot_count=int(mission.pintle_slot_count),
            slot_aspect_ratio=float(mission.pintle_slot_aspect_ratio),
            deflector_angle_deg=float(mission.pintle_deflector_angle_deg),
        ),
        feed=FeedInputs(
            architecture="electric_pump",
            fuel_tank_pressure=float(mission.P_tank_fuel),
            oxidizer_tank_pressure=float(mission.P_tank_ox),
            line_pressure_loss=float(mission.line_dp_allowance),
            pump_efficiency=float(mission.eta_pump),
            pump_speed_rpm=float(d.get("N_rpm", mission.pump_speed_rpm)),
        ),
        model_identities=identities,
        unavailable=unavailable,
    )


#: Scalars the resolved contract and a traditional ``DesignInput`` must agree
#: on exactly.  Each entry is ``(contract_path, design_input_path)``.
_DESIGN_INPUT_CROSSCHECKS: tuple[tuple[str, str], ...] = (
    ("target_thrust", "target_thrust"),
    ("chamber_pressure", "Pc"),
    ("ambient_pressure", "ambient.Pa"),
    ("contour.epsilon", "epsilon"),
    ("contour.length_pct", "length_pct"),
    ("contour.contraction_ratio", "contraction_ratio"),
    ("contour.l_star", "L_star"),
    ("contour.throat_upstream_radius_ratio",
     "throat_geometry.upstream_radius_ratio"),
    ("contour.throat_downstream_radius_ratio",
     "throat_geometry.downstream_radius_ratio"),
    ("propellant.mixture_ratio", "thermo.mixture_ratio"),
    ("performance.eta_cstar_effective", "thermo.eta_cstar"),
    ("performance.eta_CF", "thermo.eta_CF"),
    ("material.liner_conductivity", "material.conductivity"),
    ("material.liner_elastic_modulus", "material.elastic_modulus"),
    ("material.liner_thermal_expansion", "material.thermal_expansion"),
    ("material.liner_poisson_ratio", "material.poisson_ratio"),
    ("material.liner_max_gas_side_wall_temp", "material.max_temperature"),
    ("material.liner_structural_fos", "material.structural_fos"),
    ("material.liner_yield_strength", "material.yield_strength"),
    ("material.liner_density", "material.density"),
    ("thermal.channel_count", "cooling.channel_count"),
    ("thermal.channel_width", "cooling.channel_width"),
    ("thermal.channel_height", "cooling.channel_height"),
    ("thermal.channel_roughness", "cooling.channel_roughness"),
    ("thermal.regen_mass_flow", "cooling.coolant_mass_flow"),
    ("thermal.film_mass_flow", "cooling.fuel_film_mass_flow"),
    ("thermal.coolant_inlet_temperature", "cooling.coolant_inlet_temperature"),
    ("thermal.coolant_wall_temperature_limit",
     "cooling.coolant_wall_temperature_limit"),
    ("thermal.wall_thickness", "manufacturing.wall_thickness"),
    ("injector.fuel_dp_fraction", "injector.fuel_dp_fraction"),
    ("injector.oxidizer_dp_fraction", "injector.oxidizer_dp_fraction"),
    ("injector.fuel_cd", "injector.fuel_cd"),
    ("injector.oxidizer_cd", "injector.oxidizer_cd"),
    ("injector.pintle_diameter", "injector.geometry.pintle_diameter"),
    ("injector.slot_count", "injector.geometry.slot_count"),
    ("feed.fuel_tank_pressure", "injector.feed_system.fuel.tank_pressure"),
    ("feed.oxidizer_tank_pressure",
     "injector.feed_system.oxidizer.tank_pressure"),
    ("feed.pump_efficiency", "injector.feed_system.fuel.pump_efficiency"),
    ("feed.line_pressure_loss", "injector.feed_system.fuel.line_loss"),
)


def _dig(root: Any, path: str) -> Any:
    node = root
    for part in path.split("."):
        node = getattr(node, part)
    return node


def crosscheck_design_input(
    resolved: ResolvedEngineInputs,
    design_input: Any,
    *,
    rel_tol: float = 1.0e-12,
) -> tuple[str, ...]:
    """Return every scalar where a ``DesignInput`` disagrees with the contract.

    An empty tuple proves the traditional handoff carries exactly the resolved
    conventions rather than reconstructing any of them from a default.  This
    is the guard that makes switching ``to_design_input`` over to consume the
    contract a safe, verifiable step instead of a rewrite taken on faith.
    """

    drift: list[str] = []
    for contract_path, design_path in _DESIGN_INPUT_CROSSCHECKS:
        try:
            want = _dig(resolved, contract_path)
            got = _dig(design_input, design_path)
        except AttributeError as exc:
            drift.append(f"{contract_path} -> {design_path}: missing ({exc})")
            continue
        if want is None or got is None:
            if want is not got:
                drift.append(
                    f"{contract_path}={want!r} but {design_path}={got!r}"
                )
            continue
        a, b = float(want), float(got)
        if not math.isclose(a, b, rel_tol=rel_tol, abs_tol=1.0e-12):
            drift.append(
                f"{contract_path}={a!r} but {design_path}={b!r}"
            )
    return tuple(drift)
