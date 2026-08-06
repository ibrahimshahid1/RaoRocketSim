"""Versioned, pure-JAX numerical output contract for the engine MDO.

``EngineResult`` is the discipline-internal object used while evaluating the
model.  It intentionally contains Python dictionaries and rich dataclasses,
which are convenient inside a traced function but are not a stable public
output contract.  This module converts that result into nested ``NamedTuple``
pytrees containing only fixed-shape numerical leaves.

Contract rules
--------------
* Shapes depend only on :class:`~raosim.mdo.grid.GridTopology`, never on values.
* Ideal and delivered performance are both retained; efficiencies are not
  folded into ambiguous fields.
* Every profile used by cooling is retained on the same station grid.
* Battery energy and power branches remain separate.
* An unsupported quantity is ``NaN`` with a false availability entry.  The
  host-side snapshot converts that pair to ``None`` plus a reason.  A physical
  zero is never used as an availability sentinel.
* Human-readable names are module-level tuples, not JAX leaves.

Use ``jax.jit(lambda a: solve_engine_state(DesignVector.from_array(a), mission))``
with ``mission`` captured by the closure.  Passing a ``MissionSpec`` as a traced
argument is neither required nor supported.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from typing import NamedTuple

import numpy as np

import raosim.jax  # noqa: F401 -- enable float64 before constructing arrays
import jax
import jax.numpy as jnp

from raosim.mdo.assembly import StateScales
from raosim.mdo.engine import (
    EngineResult,
    chamber_surfaces_for,
    engine_residual,
    solve_engine,
)
from raosim.mdo.grid import (
    GridTopology,
    build_station_grid,
    chamber_barrel_length,
    chamber_volume,
    wetted_area,
)
from raosim.mdo.properties import ChamberSurfaces
from raosim.mdo.schema import DesignVector, MissionSpec

Array = jax.Array

ENGINE_STATE_SCHEMA_VERSION = 1


def _digest_words(digest: bytes) -> tuple[int, ...]:
    """Return a SHA-256 digest as eight stable unsigned 32-bit words."""

    return tuple(
        int.from_bytes(digest[index:index + 4], "big")
        for index in range(0, 32, 4)
    )


def mission_fingerprint_words(mission: MissionSpec) -> tuple[int, ...]:
    """Fingerprint every MissionSpec field, including strings and switches."""

    payload = json.dumps(
        asdict(mission),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    ).encode("utf-8")
    return _digest_words(hashlib.sha256(payload).digest())


def surface_fingerprint_words(surfaces: ChamberSurfaces) -> tuple[int, ...]:
    """Hash every value that defines the chamber-property evaluator.

    ``PropertySurface2D.__call__`` consumes the two grids plus ``Z``, ``Zx``,
    and ``Zy``.  Hashing only aggregate moments of ``Z`` allowed different
    grids or derivative fields to collide.  This host-side fingerprint covers
    the exact canonical float64 bytes, shapes, surface names, and bundle
    provenance.  The resulting eight words are embedded as ordinary numerical
    constants in the pure-JAX state.

    ``surfaces`` is a static host object (normally captured by the jitted solve
    closure), never a traced argument.
    """

    digest = hashlib.sha256()
    digest.update(b"raosim.ChamberSurfaces.fingerprint.v1\0")
    for label, surface in (
        ("gamma", surfaces.gamma),
        ("Tc", surfaces.Tc),
        ("R_gas", surfaces.R_gas),
    ):
        digest.update(label.encode("utf-8") + b"\0")
        digest.update(str(surface.name).encode("utf-8") + b"\0")
        for field_name in ("xg", "yg", "Z", "Zx", "Zy"):
            array = np.ascontiguousarray(
                np.asarray(getattr(surface, field_name), dtype="<f8")
            )
            digest.update(field_name.encode("utf-8") + b"\0")
            digest.update(
                json.dumps(
                    array.shape, separators=(",", ":")
                ).encode("ascii")
            )
            digest.update(b"\0")
            digest.update(array.tobytes(order="C"))
    digest.update(b"provenance\0")
    digest.update(str(surfaces.provenance).encode("utf-8"))
    return _digest_words(digest.digest())


def surface_signature(surfaces: ChamberSurfaces) -> Array:
    """Return a JIT-safe, fixed-shape fingerprint of the complete evaluator.

    Surface arrays created while tracing are JAX tracers and therefore cannot
    be copied into Python for ``hashlib``.  This uses an order-sensitive
    SplitMix64-style reduction over every float64 bit pattern, with eight
    independent lanes seeded by a SHA-256 digest of names, shapes, and
    provenance.  Eager and jitted solves consequently carry the same signature.
    """

    surface_items = (
        ("gamma", surfaces.gamma),
        ("Tc", surfaces.Tc),
        ("R_gas", surfaces.R_gas),
    )
    metadata = {
        "version": 1,
        "provenance": str(surfaces.provenance),
        "surfaces": [
            {
                "label": label,
                "name": str(surface.name),
                "shapes": {
                    field_name: tuple(
                        int(size)
                        for size in getattr(surface, field_name).shape
                    )
                    for field_name in ("xg", "yg", "Z", "Zx", "Zy")
                },
            }
            for label, surface in surface_items
        ],
    }
    seed_words = _digest_words(
        hashlib.sha256(
            json.dumps(
                metadata, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).digest()
    )
    lanes = jnp.asarray(seed_words, dtype=jnp.uint32)
    golden = jnp.asarray(0x9E3779B97F4A7C15, dtype=jnp.uint64)
    mix_a = jnp.asarray(0xBF58476D1CE4E5B9, dtype=jnp.uint64)
    mix_b = jnp.asarray(0x94D049BB133111EB, dtype=jnp.uint64)
    field_index = 0
    for _, surface in surface_items:
        for field_name in ("xg", "yg", "Z", "Zx", "Zy"):
            values = jnp.asarray(
                getattr(surface, field_name), dtype=jnp.float64
            ).reshape(-1)
            bits = jax.lax.bitcast_convert_type(
                values, jnp.uint64
            )
            positions = jnp.arange(
                1, values.size + 1, dtype=jnp.uint64
            )
            for lane_index in range(8):
                lane_seed = golden * jnp.asarray(
                    1 + lane_index + 8 * field_index,
                    dtype=jnp.uint64,
                )
                mixed = bits ^ (positions * golden) ^ lane_seed
                mixed = (mixed ^ (mixed >> 30)) * mix_a
                mixed = (mixed ^ (mixed >> 27)) * mix_b
                mixed = mixed ^ (mixed >> 31)
                folded = jnp.sum(mixed, dtype=jnp.uint64)
                word = jnp.asarray(
                    folded ^ (folded >> 32), dtype=jnp.uint32
                )
                lanes = lanes.at[lane_index].set(
                    (lanes[lane_index] ^ word)
                    * jnp.asarray(16777619, dtype=jnp.uint32)
                )
            field_index += 1
    return lanes


# Stable ordering for the numerical constraint vector.  Additions require a
# schema-version bump unless they are appended with a backwards-compatible
# reader policy.
ENGINE_CONSTRAINT_NAMES = (
    "thrust_residual",
    "separation_margin",
    "coking_margin_min",
    "land_min",
    "chug_margin_min",
    "pintle_transition_margin",
    "nss_margin_min",
    "tip_speed_margin_min",
    "aspect_ratio_margin",
    "blockage_lo_margin",
    "blockage_hi_margin",
    "structural_stress_margin",
    "wall_temp_margin",
    "film_capacity_margin",
    "property_domain_margin",
    "chart_domain_margin",
    "wall_monotonic_margin",
    "chamber_volume_margin",
    "jacket_thin_shell_margin",
    "nozzle_collapse_margin",
    "engine_residual_margin",
    "cooling_residual_margin",
    "solver_status_margin",
    "finite_margin",
)

MASS_FIELD_NAMES = (
    "pump_mass",
    "motor_mass",
    "inverter_mass",
    "battery_energy_cell_mass",
    "battery_power_cell_mass",
    "battery_energy_installed_mass",
    "battery_power_installed_mass",
    "battery_governing_installed_mass",
    "battery_objective_mass",
    "electric_feed_package_exact_mass",
    "electric_feed_package_objective_mass",
    "thrust_chamber_liner_mass",
    "thrust_chamber_land_mass",
    "thrust_chamber_closeout_mass",
    "thrust_chamber_mass",
    "injector_mass",
    "total_dry_mass",
)

# Host adapters use these stable reasons for false entries in
# ``MassState.availability``.
MASS_UNAVAILABLE_REASONS = {
    "injector_mass": (
        "The differentiable MDO sizes injector flow areas but not injector "
        "hardware mass; the injector hardware ledger needs the resolved "
        "machined layout, which is produced host-side by "
        "raosim.injector_cad.resolve_machined_pintle_layout and priced by "
        "raosim.mass_ledger.injector_mass_ledger."
    ),
    "total_dry_mass": (
        "Total dry mass is unavailable until the injector hardware mass, the "
        "chamber/injector bolted interface and the propellant-side plumbing "
        "are in the ledger; the thrust-chamber structure alone is not an "
        "engine dry mass."
    ),
}



class PerformanceState(NamedTuple):
    Pc: Array
    Pa: Array
    OF: Array
    gamma: Array
    R_gas: Array
    Tc: Array
    eta_cstar: Array
    eta_CF: Array
    eta_Isp: Array
    cstar_ideal: Array
    cstar_delivered: Array
    Cf_ideal: Array
    Cf_delivered: Array
    thrust_ideal: Array
    thrust_delivered: Array
    Isp_ideal: Array
    Isp_delivered: Array
    Ve_ideal: Array
    Ve_delivered: Array
    Rt: Array
    Re: Array
    At: Array
    Ae: Array
    epsilon: Array
    Me: Array
    Pe: Array
    Pe_over_Pc: Array
    mdot_total: Array
    mdot_oxidizer: Array
    mdot_fuel_total: Array
    mdot_fuel_core: Array
    mdot_film: Array
    mdot_regen_jacket: Array
    mdot_core_total: Array


class GeometryState(NamedTuple):
    x: Array
    r: Array
    area_ratio: Array
    mach: Array
    dseg: Array
    station_valid: Array
    throat_index: Array
    Ru_input: Array
    Rd_applied: Array
    chamber_length: Array
    convergent_length: Array
    divergent_length: Array
    # Chamber volume (injector face -> throat plane, SP-125 printed p. 88) and
    # the gas-side wetted area of the whole station grid.  Both are exported so
    # a geometry-convention drift between the two pipelines fails a parity gate
    # instead of hiding inside heat load and hardware mass.
    chamber_volume: Array
    chamber_volume_target: Array
    wetted_area: Array
    chart_domain_violation: Array
    wall_monotonic_margin: Array
    # Numeric enums keep the pytree string-free:
    # upstream_model=1 -> cosine blend; divergent_model=1 -> TOP Bézier.
    upstream_model: Array
    divergent_model: Array


class ThermalState(NamedTuple):
    T_wg: Array
    T_wc: Array
    T_coolant: Array
    T_aw: Array
    q_flux: Array
    h_g: Array
    h_c: Array
    area_enhancement: Array
    coking_margin: Array
    sigma_thermal: Array
    sigma_combined: Array
    sigma_pressure_profile: Array
    coolant_pressure: Array
    gas_pressure: Array
    liner_pressure_differential: Array
    residual: Array
    dp_total: Array
    T_coolant_exit: Array
    land_min: Array
    sigma_pressure: Array
    coolant_mach: Array
    coolant_velocity: Array
    residual_max: Array
    solver_converged: Array
    finite: Array


class InjectorState(NamedTuple):
    dp_fuel: Array
    dp_oxidizer: Array
    velocity_fuel: Array
    velocity_oxidizer: Array
    area_fuel: Array
    area_oxidizer: Array
    momentum_ratio: Array
    spray_half_angle_deg: Array
    slot_width: Array
    blockage_factor: Array
    tip_opening: Array
    area_tip_branch: Array
    area_center_gap: Array
    transition_margin: Array
    branch_consistency: Array
    chug_margin_fuel: Array
    chug_margin_oxidizer: Array


class PumpStreamState(NamedTuple):
    volumetric_flow: Array
    head: Array
    omega: Array
    rpm: Array
    specific_speed: Array
    efficiency: Array
    hydraulic_power: Array
    shaft_power: Array
    electric_power: Array
    npsh_available: Array
    suction_specific_speed: Array
    nss_margin: Array
    tip_speed: Array
    tip_speed_margin: Array
    pressure_rise: Array


class ElectricalState(NamedTuple):
    electric_power_total: Array
    battery_energy_cell_mass: Array
    battery_power_cell_mass: Array
    motor_mass: Array
    inverter_mass: Array
    pump_mass: Array


class MassState(NamedTuple):
    values: Array
    availability: Array


class ConstraintState(NamedTuple):
    values: Array
    finite: Array
    inequality_values: Array
    all_inequalities_nonnegative: Array


class ResidualState(NamedTuple):
    outer: Array
    cooling: Array
    outer_max: Array
    cooling_max: Array
    solver_status_ok: Array
    engine_solver_converged: Array
    cooling_solver_converged: Array
    finite: Array
    all_converged: Array


class InputConventionState(NamedTuple):
    """Numerical assumptions that define the solved operating-point contract.

    Human-readable identities remain host metadata, but every numerical input
    needed to establish MDO/traditional parity is carried with the JAX state.
    This prevents a solved state from being reported against an incompatible
    ``MissionSpec`` after the fact.
    """

    mission_fingerprint: Array
    surface_signature: Array
    propellant_name_code: Array
    thrust: Array
    couple_eta_cstar: Array
    ambient_pressure: Array
    burn_time: Array
    OF: Array
    eta_cstar_nominal: Array
    eta_CF: Array
    throat_ru_factor: Array
    throat_rd_factor: Array
    contraction_ratio: Array
    l_star: Array
    length_pct: Array
    cooling_fraction: Array
    coolant_temperature: Array
    rho_coolant: Array
    cp_coolant: Array
    k_coolant: Array
    mu_coolant: Array
    rho_fuel: Array
    rho_oxidizer: Array
    vapor_pressure_fuel: Array
    vapor_pressure_oxidizer: Array
    tank_pressure_fuel: Array
    tank_pressure_oxidizer: Array
    line_dp_allowance: Array
    injector_cd_fuel: Array
    injector_cd_oxidizer: Array
    pintle_slot_count: Array
    pump_speed_rpm: Array
    pump_efficiency_nominal: Array
    pump_head_coefficient: Array
    pump_tip_speed_max: Array
    pump_nss_max: Array
    motor_efficiency: Array
    inverter_efficiency: Array
    discharge_efficiency: Array
    battery_energy_density: Array
    battery_power_density: Array
    battery_structural_margin: Array
    motor_power_density: Array
    inverter_power_density: Array
    pump_specific_mass: Array
    liner_conductivity: Array
    liner_elastic_modulus: Array
    liner_thermal_expansion: Array
    liner_poisson: Array
    liner_allowable_stress: Array
    liner_structural_fos: Array
    liner_max_gas_side_temperature: Array
    channel_count: Array
    film_capacity_margin: Array
    film_system_capacity_fraction: Array


class EngineState(NamedTuple):
    schema_version: Array
    design_vector: Array
    input_conventions: InputConventionState
    performance: PerformanceState
    geometry: GeometryState
    thermal: ThermalState
    injector: InjectorState
    fuel_pump: PumpStreamState
    oxidizer_pump: PumpStreamState
    electrical: ElectricalState
    masses: MassState
    constraints: ConstraintState
    residuals: ResidualState


def _pump_state(stream, pressure_rise: Array) -> PumpStreamState:
    return PumpStreamState(
        volumetric_flow=stream.Q,
        head=stream.head,
        omega=stream.omega,
        rpm=stream.omega * 60.0 / (2.0 * jnp.pi),
        specific_speed=stream.specific_speed,
        efficiency=stream.efficiency,
        hydraulic_power=stream.P_hydraulic,
        shaft_power=stream.P_shaft,
        electric_power=stream.P_electric,
        npsh_available=stream.npsh_available,
        suction_specific_speed=stream.suction_specific_speed,
        nss_margin=stream.nss_margin,
        tip_speed=stream.tip_speed,
        tip_speed_margin=stream.tip_speed_margin,
        pressure_rise=pressure_rise,
    )


def engine_state_from_result(
    result: EngineResult,
    x: DesignVector,
    mission: MissionSpec,
    *,
    surfaces: ChamberSurfaces | None = None,
    topo: GridTopology = GridTopology(),
    couple_eta_cstar: bool = False,
) -> EngineState:
    """Convert a solved discipline result to the version-1 numerical contract.

    This function is pure array algebra and may run inside ``jit``. ``result``
    must have been produced with the same ``mission``, ``surfaces``, ``topo``,
    and coupling switch.
    """
    surfaces = surfaces if surfaces is not None else chamber_surfaces_for(mission)
    Pc = jnp.asarray(x.Pc, dtype=jnp.float64)
    eps = jnp.asarray(x.eps, dtype=jnp.float64)
    gamma = surfaces.gamma(Pc, mission.OF)
    R_gas = surfaces.R_gas(Pc, mission.OF)
    Tc = surfaces.Tc(Pc, mission.OF)
    cstar_ideal = surfaces.c_star_ideal(Pc, mission.OF)
    cstar_delivered = result.eta_cstar * cstar_ideal

    Rt = result.Rt
    Re = Rt * jnp.sqrt(eps)
    At = jnp.pi * Rt * Rt
    Ae = eps * At
    thrust_ideal = result.Cf_ideal * Pc * At
    thrust_delivered = result.Cf * Pc * At
    Isp_ideal = result.Cf_ideal * cstar_ideal / mission.g0
    Ve_ideal = Isp_ideal * mission.g0
    Ve_delivered = result.Isp * mission.g0

    mdot_fuel = result.mdot / (1.0 + mission.OF)
    mdot_oxidizer = result.mdot - mdot_fuel
    mdot_film = mdot_fuel * x.film_frac
    mdot_fuel_core = mdot_fuel - mdot_film
    mdot_jacket = mission.cooling_fraction * mdot_fuel_core
    mdot_core = mdot_oxidizer + mdot_fuel_core

    performance = PerformanceState(
        Pc=Pc,
        Pa=jnp.asarray(mission.Pa, dtype=jnp.float64),
        OF=jnp.asarray(mission.OF, dtype=jnp.float64),
        gamma=gamma,
        R_gas=R_gas,
        Tc=Tc,
        eta_cstar=result.eta_cstar,
        eta_CF=jnp.asarray(mission.eta_CF, dtype=jnp.float64),
        eta_Isp=result.eta_cstar * mission.eta_CF,
        cstar_ideal=cstar_ideal,
        cstar_delivered=cstar_delivered,
        Cf_ideal=result.Cf_ideal,
        Cf_delivered=result.Cf,
        thrust_ideal=thrust_ideal,
        thrust_delivered=thrust_delivered,
        Isp_ideal=Isp_ideal,
        Isp_delivered=result.Isp,
        Ve_ideal=Ve_ideal,
        Ve_delivered=Ve_delivered,
        Rt=Rt,
        Re=Re,
        At=At,
        Ae=Ae,
        epsilon=eps,
        Me=result.Me,
        Pe=result.Pe,
        Pe_over_Pc=result.Pe / Pc,
        mdot_total=result.mdot,
        mdot_oxidizer=mdot_oxidizer,
        mdot_fuel_total=mdot_fuel,
        mdot_fuel_core=mdot_fuel_core,
        mdot_film=mdot_film,
        mdot_regen_jacket=mdot_jacket,
        mdot_core_total=mdot_core,
    )

    grid = build_station_grid(Rt, eps, mission, topo, gamma=gamma)
    r_chamber = Rt * jnp.sqrt(mission.contraction_ratio)
    L_conv = ((r_chamber - Rt)
              / jnp.tan(jnp.deg2rad(mission.converging_half_angle_deg)))
    L_div = ((mission.length_pct / 100.0) * (Re - Rt)
             / jnp.tan(jnp.deg2rad(15.0)))
    geometry = GeometryState(
        x=grid.x,
        r=grid.r,
        area_ratio=grid.area_ratio,
        mach=grid.mach,
        dseg=grid.dseg,
        station_valid=jnp.ones_like(grid.x, dtype=jnp.bool_),
        throat_index=jnp.asarray(grid.throat_index, dtype=jnp.int32),
        Ru_input=mission.throat_ru_factor * Rt,
        Rd_applied=mission.throat_rd_factor * Rt,
        # Solved from the SP-125 chamber-volume closure (injector face to
        # throat plane = L*.A_t), not a prescribed input -- see
        # raosim.mdo.grid.chamber_barrel_length.
        chamber_length=chamber_barrel_length(Rt, mission),
        convergent_length=L_conv,
        divergent_length=L_div,
        chamber_volume=chamber_volume(Rt, mission),
        chamber_volume_target=(
            jnp.asarray(mission.l_star, dtype=jnp.float64) * jnp.pi * Rt ** 2
        ),
        wetted_area=wetted_area(grid.x, grid.r),
        chart_domain_violation=grid.chart_domain_violation,
        wall_monotonic_margin=grid.wall_monotonic_margin,
        upstream_model=jnp.asarray(1, dtype=jnp.int32),
        divergent_model=jnp.asarray(1, dtype=jnp.int32),
    )

    cool = result.cooling
    thermal = ThermalState(
        T_wg=result.T_wg,
        T_wc=cool.T_wc,
        T_coolant=cool.T_coolant,
        T_aw=cool.T_aw,
        q_flux=cool.q_flux,
        h_g=cool.h_g,
        h_c=cool.h_c,
        area_enhancement=cool.area_enh,
        coking_margin=cool.coking_margin,
        sigma_thermal=cool.sigma_thermal,
        sigma_combined=(
            cool.sigma_thermal + jnp.abs(cool.sigma_pressure_profile)
        ),
        sigma_pressure_profile=cool.sigma_pressure_profile,
        coolant_pressure=cool.coolant_pressure,
        gas_pressure=cool.gas_pressure,
        liner_pressure_differential=cool.liner_pressure_differential,
        residual=cool.residual,
        dp_total=cool.dp_total,
        T_coolant_exit=cool.T_coolant_exit,
        land_min=cool.land_min,
        sigma_pressure=cool.sigma_pressure,
        coolant_mach=cool.coolant_mach,
        coolant_velocity=cool.coolant_velocity,
        residual_max=cool.solver_residual_max,
        solver_converged=cool.solver_converged,
        finite=cool.finite,
    )

    inj = result.injector
    injector = InjectorState(
        dp_fuel=inj.dp_fuel,
        dp_oxidizer=inj.dp_ox,
        velocity_fuel=inj.v_fuel,
        velocity_oxidizer=inj.v_ox,
        area_fuel=inj.area_fuel,
        area_oxidizer=inj.area_ox,
        momentum_ratio=inj.momentum_ratio,
        spray_half_angle_deg=inj.spray_half_angle_deg,
        slot_width=inj.slot_width,
        blockage_factor=inj.blockage_factor,
        tip_opening=inj.tip_opening,
        area_tip_branch=inj.area_tip_branch,
        area_center_gap=inj.area_center_gap,
        transition_margin=inj.transition_margin,
        branch_consistency=inj.branch_consistency,
        chug_margin_fuel=inj.chug_margin_fuel,
        chug_margin_oxidizer=inj.chug_margin_ox,
    )

    fuel_pump = _pump_state(result.feed.fuel, result.dp_rise_fuel)
    oxidizer_pump = _pump_state(result.feed.ox, result.dp_rise_ox)
    electrical = ElectricalState(
        electric_power_total=result.feed.P_electric_total,
        battery_energy_cell_mass=result.feed.battery.energy_limited_mass,
        battery_power_cell_mass=result.feed.battery.power_limited_mass,
        motor_mass=result.feed.motor_mass,
        inverter_mass=result.feed.inverter_mass,
        pump_mass=result.feed.pump_mass,
    )

    batt_energy = (result.feed.battery.energy_limited_mass
                   * mission.battery_structural_margin)
    batt_power = (result.feed.battery.power_limited_mass
                  * mission.battery_structural_margin)
    batt_governing = jnp.maximum(batt_energy, batt_power)
    batt_objective = result.mass_ledger["battery_objective_smooth"]
    package_exact = (
        result.feed.pump_mass
        + result.feed.motor_mass
        + result.feed.inverter_mass
        + batt_governing
    )
    nan = jnp.asarray(jnp.nan, dtype=jnp.float64)
    chamber = result.chamber_mass
    mass_values = jnp.stack([
        result.feed.pump_mass,
        result.feed.motor_mass,
        result.feed.inverter_mass,
        result.feed.battery.energy_limited_mass,
        result.feed.battery.power_limited_mass,
        batt_energy,
        batt_power,
        batt_governing,
        batt_objective,
        package_exact,
        result.objective_mass,
        chamber.liner,
        chamber.lands,
        chamber.closeout,
        chamber.total,
        # Injector hardware and the total dry mass stay unavailable; see
        # MASS_UNAVAILABLE_REASONS.  They are NaN, never 0.0, so a consumer
        # that ignores the availability mask fails loudly instead of quietly
        # reporting weightless hardware.
        nan,
        nan,
    ])
    mass_availability = jnp.asarray(
        [True] * 15 + [False, False], dtype=jnp.bool_)
    masses = MassState(values=mass_values, availability=mass_availability)

    constraint_values = jnp.stack([
        jnp.asarray(result.constraints[name], dtype=jnp.float64)
        for name in ENGINE_CONSTRAINT_NAMES
    ])
    # The first item is an equality residual. The remaining entries are
    # conventionally margins >= 0.
    inequality_values = constraint_values[1:]
    constraints = ConstraintState(
        values=constraint_values,
        finite=jnp.all(jnp.isfinite(constraint_values)),
        inequality_values=inequality_values,
        all_inequalities_nonnegative=jnp.all(inequality_values >= 0.0),
    )

    scales = StateScales.from_mission(mission)
    y_scaled = jnp.stack([
        result.Rt / scales.Rt_ref,
        result.mdot / scales.mdot_ref,
    ])
    outer = engine_residual(
        y_scaled, x, mission, surfaces, scales, couple_eta_cstar)
    residual_finite = (jnp.all(jnp.isfinite(outer))
                       & jnp.all(jnp.isfinite(cool.residual))
                       & result.finite)
    residuals = ResidualState(
        outer=outer,
        cooling=cool.residual,
        outer_max=jnp.max(jnp.abs(outer)),
        cooling_max=jnp.max(jnp.abs(cool.residual)),
        solver_status_ok=result.solver_status_ok,
        engine_solver_converged=result.solver_converged,
        cooling_solver_converged=cool.solver_converged,
        finite=residual_finite,
        all_converged=(result.solver_converged
                       & cool.solver_converged & residual_finite),
    )

    input_conventions = InputConventionState(
        mission_fingerprint=jnp.asarray(
            mission_fingerprint_words(mission), dtype=jnp.uint32
        ),
        surface_signature=surface_signature(surfaces),
        propellant_name_code=jnp.asarray(
            int.from_bytes(
                hashlib.sha256(
                    str(mission.propellant_name)
                    .strip()
                    .lower()
                    .encode("utf-8")
                ).digest()[:4],
                "big",
            ),
            dtype=jnp.uint32,
        ),
        thrust=jnp.asarray(mission.thrust, dtype=jnp.float64),
        couple_eta_cstar=jnp.asarray(couple_eta_cstar, dtype=jnp.bool_),
        ambient_pressure=jnp.asarray(mission.Pa, dtype=jnp.float64),
        burn_time=jnp.asarray(mission.burn_time, dtype=jnp.float64),
        OF=jnp.asarray(mission.OF, dtype=jnp.float64),
        eta_cstar_nominal=jnp.asarray(mission.eta_cstar, dtype=jnp.float64),
        eta_CF=jnp.asarray(mission.eta_CF, dtype=jnp.float64),
        throat_ru_factor=jnp.asarray(
            mission.throat_ru_factor, dtype=jnp.float64
        ),
        throat_rd_factor=jnp.asarray(
            mission.throat_rd_factor, dtype=jnp.float64
        ),
        contraction_ratio=jnp.asarray(
            mission.contraction_ratio, dtype=jnp.float64
        ),
        l_star=jnp.asarray(mission.l_star, dtype=jnp.float64),
        length_pct=jnp.asarray(mission.length_pct, dtype=jnp.float64),
        cooling_fraction=jnp.asarray(
            mission.cooling_fraction, dtype=jnp.float64
        ),
        coolant_temperature=jnp.asarray(
            mission.coolant_temperature, dtype=jnp.float64
        ),
        rho_coolant=jnp.asarray(mission.rho_cool, dtype=jnp.float64),
        cp_coolant=jnp.asarray(mission.cp_cool, dtype=jnp.float64),
        k_coolant=jnp.asarray(mission.k_cool, dtype=jnp.float64),
        mu_coolant=jnp.asarray(mission.mu_cool, dtype=jnp.float64),
        rho_fuel=jnp.asarray(mission.rho_fuel, dtype=jnp.float64),
        rho_oxidizer=jnp.asarray(mission.rho_ox, dtype=jnp.float64),
        vapor_pressure_fuel=jnp.asarray(
            mission.p_vapor_fuel, dtype=jnp.float64
        ),
        vapor_pressure_oxidizer=jnp.asarray(
            mission.p_vapor_ox, dtype=jnp.float64
        ),
        tank_pressure_fuel=jnp.asarray(
            mission.P_tank_fuel, dtype=jnp.float64
        ),
        tank_pressure_oxidizer=jnp.asarray(
            mission.P_tank_ox, dtype=jnp.float64
        ),
        line_dp_allowance=jnp.asarray(
            mission.line_dp_allowance, dtype=jnp.float64
        ),
        injector_cd_fuel=jnp.asarray(
            mission.injector_cd_fuel, dtype=jnp.float64
        ),
        injector_cd_oxidizer=jnp.asarray(
            mission.injector_cd_ox, dtype=jnp.float64
        ),
        pintle_slot_count=jnp.asarray(
            mission.pintle_slot_count, dtype=jnp.int32
        ),
        pump_speed_rpm=jnp.asarray(x.N_rpm, dtype=jnp.float64),
        pump_efficiency_nominal=jnp.asarray(
            mission.eta_pump, dtype=jnp.float64
        ),
        pump_head_coefficient=jnp.asarray(
            mission.pump_head_coefficient, dtype=jnp.float64
        ),
        pump_tip_speed_max=jnp.asarray(
            mission.pump_tip_speed_max, dtype=jnp.float64
        ),
        pump_nss_max=jnp.asarray(mission.pump_nss_max, dtype=jnp.float64),
        motor_efficiency=jnp.asarray(mission.eta_motor, dtype=jnp.float64),
        inverter_efficiency=jnp.asarray(
            mission.eta_inverter, dtype=jnp.float64
        ),
        discharge_efficiency=jnp.asarray(
            mission.eta_discharge, dtype=jnp.float64
        ),
        battery_energy_density=jnp.asarray(
            mission.battery_energy_density, dtype=jnp.float64
        ),
        battery_power_density=jnp.asarray(
            mission.battery_power_density, dtype=jnp.float64
        ),
        battery_structural_margin=jnp.asarray(
            mission.battery_structural_margin, dtype=jnp.float64
        ),
        motor_power_density=jnp.asarray(
            mission.motor_power_density, dtype=jnp.float64
        ),
        inverter_power_density=jnp.asarray(
            mission.inverter_power_density, dtype=jnp.float64
        ),
        pump_specific_mass=jnp.asarray(
            mission.pump_specific_mass, dtype=jnp.float64
        ),
        liner_conductivity=jnp.asarray(mission.k_wall, dtype=jnp.float64),
        liner_elastic_modulus=jnp.asarray(mission.liner_E, dtype=jnp.float64),
        liner_thermal_expansion=jnp.asarray(
            mission.liner_alpha, dtype=jnp.float64
        ),
        liner_poisson=jnp.asarray(
            mission.liner_poisson, dtype=jnp.float64
        ),
        liner_allowable_stress=jnp.asarray(
            mission.liner_sigma_allow, dtype=jnp.float64
        ),
        liner_structural_fos=jnp.asarray(
            mission.liner_structural_fos, dtype=jnp.float64
        ),
        liner_max_gas_side_temperature=jnp.asarray(
            mission.liner_T_wg_max, dtype=jnp.float64
        ),
        channel_count=jnp.asarray(mission.n_channels, dtype=jnp.int32),
        film_capacity_margin=jnp.asarray(
            mission.film_capacity_margin, dtype=jnp.float64
        ),
        film_system_capacity_fraction=jnp.asarray(
            mission.film_system_capacity_fraction, dtype=jnp.float64
        ),
    )

    return EngineState(
        schema_version=jnp.asarray(
            ENGINE_STATE_SCHEMA_VERSION, dtype=jnp.int32),
        design_vector=x.to_array(),
        input_conventions=input_conventions,
        performance=performance,
        geometry=geometry,
        thermal=thermal,
        injector=injector,
        fuel_pump=fuel_pump,
        oxidizer_pump=oxidizer_pump,
        electrical=electrical,
        masses=masses,
        constraints=constraints,
        residuals=residuals,
    )


def solve_engine_state(
    x: DesignVector,
    mission: MissionSpec,
    *,
    couple_eta_cstar: bool = False,
    surfaces: ChamberSurfaces | None = None,
    topo: GridTopology = GridTopology(),
    rtol: float = 1e-12,
    atol: float = 1e-12,
    max_steps: int = 64,
) -> EngineState:
    """Solve the engine and return the versioned pure-JAX state pytree."""
    surfaces = surfaces if surfaces is not None else chamber_surfaces_for(mission)
    result = solve_engine(
        x,
        mission,
        couple_eta_cstar=couple_eta_cstar,
        surfaces=surfaces,
        topo=topo,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
    )
    return engine_state_from_result(
        result,
        x,
        mission,
        surfaces=surfaces,
        topo=topo,
        couple_eta_cstar=couple_eta_cstar,
    )


__all__ = [
    "ENGINE_STATE_SCHEMA_VERSION",
    "ENGINE_CONSTRAINT_NAMES",
    "MASS_FIELD_NAMES",
    "MASS_UNAVAILABLE_REASONS",
    "PerformanceState",
    "GeometryState",
    "ThermalState",
    "InjectorState",
    "PumpStreamState",
    "ElectricalState",
    "MassState",
    "ConstraintState",
    "ResidualState",
    "InputConventionState",
    "EngineState",
    "mission_fingerprint_words",
    "surface_fingerprint_words",
    "surface_signature",
    "engine_state_from_result",
    "solve_engine_state",
]
