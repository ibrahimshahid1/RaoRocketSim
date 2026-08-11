"""Post-optimum bridge from the differentiable MDO to authoritative outputs.

This module is intentionally outside the differentiated path.  It maps an MDO
design onto the traditional ``DesignInput`` contract, runs
``design_nozzle_v2``, optionally sizes the electric pump package from the
resulting feed ledger, and compares versioned host-side snapshots.

The bridge keeps three rules explicit:

* shared inputs and conventions are mapped, not silently defaulted;
* unavailable physics (notably film-injector hardware and complete engine mass)
  remains unavailable rather than becoming a fake zero;
* the full authoritative result and generated artifacts remain attached to the
  returned re-evaluation object.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from raosim.mdo.schema import (
    DesignLayout,
    MissionSpec,
    validate_mixture_ratio,
)
from raosim.mdo.snapshot import (
    EngineAnalysisSnapshot,
    SnapshotComparison,
    _validate_engine_state_design,
    _validate_engine_state_mission,
    _validate_engine_state_schema,
    compare_snapshots,
    snapshot_from_mdo,
    snapshot_from_traditional,
)


def _as_design_dict(design: Any) -> dict[str, Any]:
    if isinstance(design, Mapping):
        return dict(design)
    if hasattr(design, "as_dict"):
        return dict(design.as_dict())
    raise TypeError("design must be a mapping or DesignVector")


def _propellant_mapping(
    mission: MissionSpec,
    propellant: str | None,
    coolant: str | None,
) -> tuple[str, str | None, str | None, str]:
    """Return combination, oxidizer, fuel, and coolant names."""

    combination = str(propellant or mission.propellant_name)
    oxidizer = fuel = None
    if "/" in combination:
        oxidizer, fuel = (part.strip() for part in combination.split("/", 1))
    coolant_name = coolant
    if coolant_name is None:
        try:
            from raosim.mdo.propellants import get_propellant

            coolant_name = get_propellant(combination).coolant_name
        except Exception:
            coolant_name = fuel
    if not coolant_name:
        raise ValueError(
            "could not infer coolant identity from the MDO propellant; pass coolant="
        )
    return combination, oxidizer, fuel, str(coolant_name)


def _effective_of(
    design: Mapping[str, Any],
    mission: MissionSpec,
    *,
    layout: DesignLayout | None = None,
    solved_of: Any | None = None,
) -> float:
    """Mixture ratio the MDO design actually used.

    A solved result is authoritative when supplied.  Otherwise the explicit
    layout selects either the live design value or the mission's fixed value.
    Merely finding an ``OF`` key is never treated as proof that O/F was active;
    this is what prevents legacy fixed-mode sentinels from leaking into a flow
    split.
    """

    if solved_of is not None:
        return validate_mixture_ratio(solved_of, name="solved EngineResult.OF")
    resolved_layout = layout or mission.design_layout()
    if resolved_layout.of_is_variable:
        if "OF" not in design or design["OF"] is None:
            raise ValueError(
                "variable-O/F design is missing its optimized physical OF value"
            )
        return validate_mixture_ratio(design["OF"], name="design.OF")
    return validate_mixture_ratio(mission.OF, name="MissionSpec.OF")


def _mdo_mass_flow(design: Mapping[str, Any], mission: MissionSpec) -> float:
    """Reproduce the MDO thrust/c-star closure without a hard-coded Cf."""

    from raosim.gas_dynamics import (
        isentropic_pressure_ratio,
        mach_from_area_ratio,
        thrust_coefficient,
    )

    Pc = float(design["Pc"])
    eps = float(design["eps"])
    gamma = float(mission.gamma)
    Me = mach_from_area_ratio(eps, gamma, supersonic=True)
    pe_pc = isentropic_pressure_ratio(Me, gamma)
    cf_ideal = thrust_coefficient(
        Me, gamma, pe_pc, float(mission.Pa) / Pc, eps
    )
    eta_cf = float(getattr(mission, "eta_CF", 1.0))
    cf_delivered = cf_ideal * eta_cf
    At = float(mission.thrust) / (cf_delivered * Pc)
    film_frac = float(design.get("film_frac", 0.0))
    eta_cstar = float(mission.eta_cstar) * (
        1.0 - float(mission.film_cstar_penalty) * film_frac
    )
    return Pc * At / (eta_cstar * float(mission.c_star_ideal()))


# --------------------------------------------------------------------------- #
# MDO result -> traditional DesignInput                                        #
# --------------------------------------------------------------------------- #
def to_design_input(
    design: Any,
    mission: MissionSpec,
    *,
    mdot_cool: float | None = None,
    mdot_total: float | None = None,
    propellant: str | None = None,
    coolant: str | None = None,
    mode: str | None = None,
    thermo_mode: str | None = None,
    contour_method: str = "bezier",
    eta_cstar: float | None = None,
    eta_CF: float | None = None,
    material_name: str | None = None,
    output_dir: str | Path | None = None,
    cad: str = "none",
    csv_points: int = 301,
    angular_points: int = 64,
    host_rao_solver_options: Mapping[str, Any] | None = None,
    pinned_chamber_state: Any | None = None,
    effective_of: float | None = None,
) -> Any:
    """Build a convention-aligned traditional ``DesignInput``.

    The map carries the actual propellant/coolant identities, O/F, split
    c-star/Cf efficiency convention, ambient pressure, channel and material assumptions,
    both injector pressure-drop fractions and discharge coefficients, throat
    radius ratios, and the pump/tank pressure ledger.  ``mdot_cool`` means the
    jacket flow *after* the MDO's explicit film diversion.

    A non-zero MDO film fraction is carried as an explicit branch downstream of
    the common fuel pump.  The regenerative-jacket and film flows must close to
    total cycle fuel; film-injector hardware remains explicitly unsupported.
    """

    from raosim.design import (
        CoolingSpec,
        DESIGN_MODE_PRELIMINARY,
        DesignInput,
        HostRaoSolverSpec,
        ManufacturingSpec,
        MaterialSpec,
        MissionAmbientSpec,
        ThermoSpec,
    )
    from raosim.cea import THERMO_PINNED_CHAMBER
    from raosim.injector import (
        FeedLineSpec,
        FeedSystemSpec,
        InjectorSpec,
        PintleGeometrySpec,
        PropellantFeedSpec,
        resolve_feed_state,
    )
    from raosim.throat_geometry import ThroatGeometrySpec

    d = _as_design_dict(design)
    combination, oxidizer, fuel, coolant_name = _propellant_mapping(
        mission, propellant, coolant
    )
    eta_cf = float(
        eta_CF if eta_CF is not None else getattr(mission, "eta_CF", 1.0)
    )
    film_frac = float(d.get("film_frac", 0.0))
    resolved_of = _effective_of(
        d,
        mission,
        solved_of=effective_of,
    )
    eta_cstar_effective = (
        float(eta_cstar)
        if eta_cstar is not None
        else float(mission.eta_cstar)
        * (1.0 - float(mission.film_cstar_penalty) * film_frac)
    )

    if mdot_total is None:
        mdot_total = _mdo_mass_flow(d, mission)
    mdot_fuel = float(mdot_total) / (1.0 + resolved_of)
    if mdot_cool is None:
        # Explicit architecture assumption in the MDO: the selected fraction
        # is diverted from the jacket to a separate fuel-film path.
        mdot_cool = (
            float(mission.cooling_fraction)
            * mdot_fuel
            * (1.0 - film_frac)
        )
    mdot_film = mdot_fuel * film_frac
    if not math.isclose(
        float(mdot_cool) + mdot_film,
        mdot_fuel,
        rel_tol=1.0e-9,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "the authoritative regen+film topology requires jacket flow + "
            "film flow = total fuel flow; MissionSpec.cooling_fraction values "
            "below 1 need an explicit third fuel-bypass branch before parity "
            "post-processing"
        )

    Pc = float(d["Pc"])
    fuel_outlet_pressure = Pc * (1.0 + float(d["dp_f_frac"]))
    # The MDO currently uses constant liquid feed properties even when a
    # screening thermal march predicts a very hot jacket outlet.  Make that
    # approximation explicit in the authoritative handoff by completing the
    # property tuple (surface tension and oxidizer viscosity are not MDO
    # variables) with the traditional resolver at its documented storage state.
    # A full tuple prevents a backend/version-dependent partial override.
    fuel_base = resolve_feed_state(
        PropellantFeedSpec(role="fuel", name=fuel),
        default_pressure=float(mission.P_tank_fuel),
    )
    oxidizer_base = resolve_feed_state(
        PropellantFeedSpec(role="oxidizer", name=oxidizer),
        default_pressure=float(mission.P_tank_ox),
    )
    if (
        pinned_chamber_state is not None
        and thermo_mode is not None
        and thermo_mode != THERMO_PINNED_CHAMBER
    ):
        raise ValueError(
            "pinned_chamber_state cannot be combined with a different "
            "thermo_mode"
        )
    resolved_thermo_mode = (
        THERMO_PINNED_CHAMBER
        if pinned_chamber_state is not None
        else (
            thermo_mode
            if thermo_mode is not None
            else ("cea_frozen" if mission.cea_table_path else "constant_gamma")
        )
    )
    thermo = ThermoSpec(
        mode=resolved_thermo_mode,
        propellant_name=combination,
        oxidizer=oxidizer,
        fuel=fuel,
        mixture_ratio=resolved_of,
        eta_Isp=eta_cstar_effective * eta_cf,
        eta_cstar=eta_cstar_effective,
        eta_CF=eta_cf,
        pinned_chamber_state=pinned_chamber_state,
    )
    cooling_spec = CoolingSpec(
        method="regenerative",
        coolant=coolant_name,
        channel_count=int(mission.n_channels),
        channel_width=float(d["channel_width"]),
        channel_height=float(d["channel_height"]),
        channel_roughness=float(mission.channel_roughness),
        coolant_mass_flow=float(mdot_cool),
        fuel_film_mass_flow=float(mdot_film),
        coolant_cp=float(mission.cp_cool),
        coolant_inlet_temperature=float(mission.coolant_temperature),
        # jacket -> fuel injector boundary, independent of jacket dP
        coolant_outlet_pressure=fuel_outlet_pressure,
        injector_pressure_drop=0.0,
        max_wall_temperature=float(mission.liner_T_wg_max),
        coolant_density=float(mission.rho_cool),
        coolant_viscosity=float(mission.mu_cool),
        coolant_conductivity=float(mission.k_cool),
        coolant_wall_temperature_limit=float(
            mission.rp1_coking_wall_temp_K
        ),
        coolant_property_backend="constant",
    )
    material = MaterialSpec(
        name=(
            material_name
            or getattr(mission, "liner_material_name", None)
            # No catalog record backs the class-default wall constants, so the
            # handoff must not name an alloy the MDO never traced.
            or "unattributed_class_default"
        ),
        yield_strength=(
            float(mission.liner_sigma_allow)
            * float(mission.liner_structural_fos)
        ),
        conductivity=float(mission.k_wall),
        max_temperature=float(mission.liner_T_wg_max),
        # The MDO has no independent material heat-flux allowable.  Infinity
        # explicitly disables that unmatched gate; thermal limits still act
        # through wall temperature and combined stress.
        max_heat_flux=float(getattr(mission, "liner_heat_flux_max", math.inf)),
        elastic_modulus=float(mission.liner_E),
        thermal_expansion=float(mission.liner_alpha),
        poisson_ratio=float(mission.liner_poisson),
        structural_fos=float(mission.liner_structural_fos),
        # ``rho_wall`` is the density the MDO's own station-grid mass integral
        # uses (raosim.mdo.mass).  Handing it across means the authoritative
        # re-evaluation prices the *same* metal, so a chamber-mass difference
        # between the two paths can only come from geometry or thickness --
        # never from two pipelines silently assuming different alloys.
        density=(
            getattr(mission, "liner_density", None)
            if getattr(mission, "liner_density", None) is not None
            else float(getattr(mission, "rho_wall", 0.0)) or None
        ),
        category="copper_alloy_screening",
    )
    manufacturing = ManufacturingSpec(
        wall_thickness=float(d["t_wall"]),
        cad=str(cad),
        output_dir=(Path(output_dir) if output_dir is not None else None),
        csv_points=int(csv_points),
        angular_points=int(angular_points),
    )
    feed = FeedSystemSpec(
        architecture="pump_fed",
        fuel=FeedLineSpec(
            line_loss=float(mission.line_dp_allowance),
            tank_pressure=float(mission.P_tank_fuel),
            pump_efficiency=float(mission.eta_pump),
        ),
        oxidizer=FeedLineSpec(
            line_loss=float(mission.line_dp_allowance),
            tank_pressure=float(mission.P_tank_ox),
            pump_efficiency=float(mission.eta_pump),
        ),
    )
    injector = InjectorSpec(
        type="pintle",
        # The MDO uses discrete radial slots and auto-sizes their flow area.
        # Its separate Son transition inequality remains a screening field;
        # selecting the traditional continuous-gap architecture would remove
        # the slots and blockage-factor geometry being compared.
        architecture="fixed_discrete",
        sizing="auto",
        fuel_dp_fraction=float(d["dp_f_frac"]),
        oxidizer_dp_fraction=float(d["dp_o_frac"]),
        fuel_cd=float(mission.injector_cd_fuel),
        oxidizer_cd=float(mission.injector_cd_ox),
        fuel=PropellantFeedSpec(
            role="fuel",
            name=fuel,
            # The regenerative jacket owns the injector-inlet temperature.
            # Supplying the storage-state temperature here creates a false
            # topology conflict when the traditional cooling solve hands off
            # its resolved outlet.  All constant-liquid properties needed by
            # the MDO parity convention remain explicit below.
            inlet_temperature=None,
            phase="liquid",
            density=float(mission.rho_fuel),
            viscosity=float(mission.mu_cool),
            surface_tension=float(fuel_base.surface_tension),
            vapor_pressure=float(mission.p_vapor_fuel),
            property_source=(
                "constant-liquid parity input: MDO MissionSpec "
                "density/viscosity/vapor pressure; traditional documented "
                "storage-state surface tension"
            ),
        ),
        oxidizer=PropellantFeedSpec(
            role="oxidizer",
            name=oxidizer,
            inlet_temperature=float(oxidizer_base.temperature),
            phase="liquid",
            density=float(mission.rho_ox),
            viscosity=float(oxidizer_base.viscosity),
            surface_tension=float(oxidizer_base.surface_tension),
            vapor_pressure=float(mission.p_vapor_ox),
            property_source=(
                "constant-liquid parity input: MDO MissionSpec density/vapor "
                "pressure; traditional documented storage-state viscosity/"
                "surface tension"
            ),
        ),
        allow_infeasible=True,
        geometry=PintleGeometrySpec(
            pintle_diameter=float(d["D_pintle"]),
            slot_count=int(mission.pintle_slot_count),
            slot_aspect_ratio=float(mission.pintle_slot_aspect_ratio),
            deflector_angle=float(mission.pintle_deflector_angle_deg),
            radial_stream="fuel",
            radial_exit_style="slots",
        ),
        feed_system=feed,
    )
    throat = ThroatGeometrySpec(
        upstream_radius_ratio=float(getattr(mission, "throat_ru_factor", 1.5)),
        downstream_radius_ratio=float(
            getattr(mission, "throat_rd_factor", 0.382)
        ),
        convergent_half_angle_deg=float(mission.converging_half_angle_deg),
    )
    return DesignInput(
        thermo=thermo,
        Pc=Pc,
        target_thrust=float(mission.thrust),
        epsilon=float(d["eps"]),
        method=str(contour_method),
        mode=mode or DESIGN_MODE_PRELIMINARY,
        length_pct=float(mission.length_pct),
        contraction_ratio=float(mission.contraction_ratio),
        L_star=float(mission.l_star),
        throat_geometry=throat,
        ambient=MissionAmbientSpec(Pa=float(mission.Pa)),
        cooling=cooling_spec,
        material=material,
        manufacturing=manufacturing,
        injector=injector,
        host_rao_solver=HostRaoSolverSpec(
            **dict(host_rao_solver_options or {})
        ),
    )


def _feed_ledger_from_result(result: Any) -> Any | None:
    """Rehydrate the report's feed-ledger dict for ``size_electric_pumps``."""

    from raosim.injector import FeedLineLedger, FeedSystemLedger

    injector = result.report_sections.get("injector", {})
    if not isinstance(injector, Mapping):
        return None
    raw = injector.get("feed_system")
    if not isinstance(raw, Mapping) or not isinstance(raw.get("lines"), Mapping):
        return None
    lines = {}
    key_map = {
        "chamber_pressure": "chamber_pressure_pa",
        "injector_dp": "injector_dp_pa",
        "manifold_loss": "manifold_loss_pa",
        "manifold_screen_loss": "manifold_screen_loss_pa",
        "regen_loss": "regen_loss_pa",
        "line_valve_loss": "line_valve_loss_pa",
        "control_margin": "control_margin_pa",
        "required_outlet_pressure": "required_outlet_pressure_pa",
        "available_outlet_pressure": "available_outlet_pressure_pa",
        "pressure_margin": "pressure_margin_pa",
        "density": "density_kg_m3",
        "viscosity": "viscosity_pa_s",
        "vapor_pressure": "vapor_pressure_pa",
        "volumetric_flow": "volumetric_flow_m3_s",
        "required_pressure_rise": "required_pressure_rise_pa",
        "required_pump_head": "required_pump_head_m",
        "ideal_pump_power": "ideal_pump_power_w",
        "flow_capacity": "flow_capacity_kg_s",
        "capacity_margin": "capacity_margin_kg_s",
        "npsh_available": "npsh_available_pa",
        "npsh_required": "npsh_required_pa",
        "npsh_margin": "npsh_margin_pa",
        "status": "status",
    }
    for role, line in raw["lines"].items():
        if not isinstance(line, Mapping):
            return None
        kwargs = {"role": str(role)}
        for attr, key in key_map.items():
            kwargs[attr] = line.get(key)
        lines[str(role)] = FeedLineLedger(**kwargs)
    return FeedSystemLedger(
        architecture=str(raw.get("architecture", "pump_fed")),
        lines=lines,
        governing_required_pressure=float(
            raw.get("governing_required_pressure_pa", 0.0)
        ),
        notes=list(raw.get("notes", ())),
    )


def _usable_feed_ledger(ledger: Any | None) -> bool:
    if ledger is None or not getattr(ledger, "lines", None):
        return False
    for role in ("fuel", "oxidizer"):
        line = ledger.lines.get(role)
        if line is None:
            return False
        for value in (
            line.volumetric_flow,
            line.required_pressure_rise,
            line.required_pump_head,
        ):
            if value is None or not np.isfinite(float(value)):
                return False
    return True


def _pump_spec(
    design: Mapping[str, Any],
    mission: MissionSpec,
    mdo_result: Any | None,
) -> Any:
    from raosim.pumps import (
        BatterySpec,
        ElectricDriveSpec,
        PumpSizingSpec,
    )

    efficiencies: dict[str, float | None] = {}
    if mdo_result is not None:
        if hasattr(mdo_result, "fuel_pump"):
            efficiencies = {
                "fuel": float(mdo_result.fuel_pump.efficiency),
                "oxidizer": float(mdo_result.oxidizer_pump.efficiency),
            }
        else:
            efficiencies = {
                "fuel": float(mdo_result.feed.fuel.efficiency),
                "oxidizer": float(mdo_result.feed.ox.efficiency),
            }
    else:
        efficiencies = {"default": float(mission.eta_pump)}
    return PumpSizingSpec(
        drive=ElectricDriveSpec(
            motor_efficiency=float(mission.eta_motor),
            inverter_efficiency=float(mission.eta_inverter),
            rpm=float(design["N_rpm"]),
            motor_power_density=float(mission.motor_power_density),
            inverter_power_density=float(mission.inverter_power_density),
        ),
        battery=BatterySpec(
            energy_density=float(mission.battery_energy_density),
            power_density=float(mission.battery_power_density),
            discharge_efficiency=float(mission.eta_discharge),
            structural_margin=float(mission.battery_structural_margin),
        ),
        burn_time=float(mission.burn_time),
        pump_efficiency=efficiencies,
        head_coefficient=float(mission.pump_head_coefficient),
        material_tip_speed_limit=float(mission.pump_tip_speed_max),
    )


def _snapshot_report_path(design_input: Any) -> Path | None:
    """Return the authoritative contract-report path when export is enabled."""

    output_dir = getattr(design_input.manufacturing, "output_dir", None)
    if output_dir is None:
        return None
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out / "engine_analysis_snapshot_v1.json"


def _write_snapshot_report(
    path: Path,
    *,
    authoritative: EngineAnalysisSnapshot,
    mdo: EngineAnalysisSnapshot | None,
    comparison: SnapshotComparison | None,
) -> None:
    """Write the authoritative post-optimum report/CAD handoff.

    The traditional snapshot is deliberately the primary payload.  The MDO
    state and parity comparison are attached as screening evidence, never as
    replacements for the authoritative contour, gates, or generated artifacts.
    """

    payload = {
        "authoritative": authoritative.to_dict(),
        "mdo_screening": mdo.to_dict() if mdo is not None else None,
        "comparison": comparison.to_dict() if comparison is not None else None,
    }
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _finite_json_metadata(value: Any) -> Any:
    """Sanitize parsed report metadata for standards-compliant JSON."""

    if isinstance(value, Mapping):
        return {
            str(key): _finite_json_metadata(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_finite_json_metadata(item) for item in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(
        float(value)
    ):
        return None
    if isinstance(value, np.generic):
        return _finite_json_metadata(value.item())
    return value


def _attach_snapshot_handoff(
    result: Any,
    snapshot_path: Path,
    authoritative: EngineAnalysisSnapshot,
) -> tuple[str, ...]:
    """Point generated report/CAD metadata at the authoritative contract.

    ``design_nozzle_v2`` must finish before a traditional snapshot can be
    constructed.  Therefore its JSON report and CAD sidecars are augmented in
    this host-side post-processing step, after the immutable snapshot report is
    written.  Geometry files themselves are never rewritten.
    """

    digest = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    handoff = {
        "role": "authoritative_post_optimization_analysis",
        "snapshot_path": str(snapshot_path),
        "snapshot_sha256": digest,
        "contract_name": authoritative.contract_name,
        "contract_version": authoritative.contract_version,
        "analysis_source": authoritative.source,
        "optimizer_metadata": dict(authoritative.optimizer_metadata or {}),
    }
    warnings: list[str] = []
    seen: set[Path] = set()
    for name, raw_path in dict(getattr(result, "files", {})).items():
        path = Path(raw_path)
        # Every JSON artifact emitted by design_nozzle_v2 is a downstream
        # report/CAD metadata consumer.  Matching file-dictionary key suffixes
        # missed real outputs such as ``pintle_parameters_json``.
        if path.suffix.lower() != ".json" or not path.is_file():
            continue
        resolved = path.resolve()
        if resolved == snapshot_path.resolve() or resolved in seen:
            continue
        seen.add(resolved)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("top-level JSON payload is not an object")
            payload["authoritative_analysis_snapshot"] = handoff
            path.write_text(
                json.dumps(
                    _finite_json_metadata(payload),
                    indent=2,
                    allow_nan=False,
                ) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            warnings.append(
                f"could not attach authoritative snapshot metadata to "
                f"{name} ({path}): {type(exc).__name__}: {exc}"
            )
    return tuple(warnings)


def _solve_mdo_for_snapshot(
    design: Mapping[str, Any],
    mission: MissionSpec,
    *,
    couple_eta_cstar: bool = False,
    surfaces: Any | None = None,
) -> Any:
    from raosim.mdo.schema import DesignVector
    from raosim.mdo.state import solve_engine_state

    layout = mission.design_layout()
    contract = dict(design)
    contract["OF"] = _effective_of(contract, mission, layout=layout)
    values = [contract[name] for name in DesignVector.names()]
    return solve_engine_state(
        DesignVector.from_contract_array(
            np.asarray(values, dtype=float), layout
        ),
        mission,
        couple_eta_cstar=couple_eta_cstar,
        surfaces=surfaces,
    )


def _mdo_total_mass_flow(mdo_result: Any) -> float:
    if hasattr(mdo_result, "performance") and hasattr(
        mdo_result.performance, "mdot_total"
    ):
        return float(mdo_result.performance.mdot_total)
    return float(mdo_result.mdot)


def _mdo_eta_cstar(mdo_result: Any) -> float:
    if hasattr(mdo_result, "performance") and hasattr(
        mdo_result.performance, "eta_cstar"
    ):
        return float(mdo_result.performance.eta_cstar)
    return float(mdo_result.eta_cstar)


def _mdo_effective_of(mdo_result: Any) -> float:
    """Return the authoritative O/F retained by EngineState/EngineResult."""

    if hasattr(mdo_result, "performance") and hasattr(
        mdo_result.performance, "OF"
    ):
        value = mdo_result.performance.OF
    elif hasattr(mdo_result, "OF"):
        value = mdo_result.OF
    else:
        raise ValueError("MDO result does not retain an authoritative OF")
    return validate_mixture_ratio(value, name="MDO result.OF")


def _pinned_chamber_state_from_mdo(
    mdo_result: Any,
    mission: MissionSpec,
    *,
    surfaces: Any | None = None,
) -> Any | None:
    """Return the exact calorically-perfect chamber state solved by the MDO.

    Legacy ``EngineResult`` objects do not retain gamma/Tc/R/c-star and return
    ``None``.  Versioned ``EngineState`` objects retain all four values plus the
    complete property-surface fingerprint, so the host pipeline can consume the
    same thermochemistry instead of launching a second, potentially different
    CEA/fallback calculation.
    """

    performance = getattr(mdo_result, "performance", None)
    conventions = getattr(mdo_result, "input_conventions", None)
    required = ("gamma", "Tc", "R_gas", "cstar_ideal")
    if performance is None or not all(
        hasattr(performance, name) for name in required
    ):
        return None

    from raosim.cea import PinnedChamberState

    fingerprint = None
    if conventions is not None and hasattr(conventions, "surface_signature"):
        words = np.asarray(
            conventions.surface_signature, dtype=np.uint32
        ).reshape(-1)
        fingerprint = "".join(f"{int(word):08x}" for word in words)
    surface_provenance = getattr(surfaces, "provenance", None)
    if surface_provenance is None:
        surface_provenance = (
            f"cea_frozen_table:{mission.cea_table_path}"
            if mission.cea_table_path
            else "constant_fallback_from_MissionSpec"
        )
    return PinnedChamberState(
        gamma=float(performance.gamma),
        Tc=float(performance.Tc),
        R_gas=float(performance.R_gas),
        c_star_ideal=float(performance.cstar_ideal),
        source=(
            "pinned_from_mdo_EngineState:"
            + str(surface_provenance)
        ),
        surface_fingerprint=fingerprint,
    )


def _validate_mdo_state_handoff(
    mdo_result: Any,
    design: Mapping[str, Any],
    mission: MissionSpec,
    *,
    couple_eta_cstar: bool,
    surfaces: Any | None = None,
) -> None:
    """Validate state/schema/design/mission identity and coupling convention."""

    if not (
        hasattr(mdo_result, "schema_version")
        and hasattr(mdo_result, "design_vector")
        and hasattr(mdo_result, "input_conventions")
    ):
        return
    _validate_engine_state_schema(mdo_result)
    _validate_engine_state_design(mdo_result, design)
    _validate_engine_state_mission(
        mdo_result,
        mission,
        surfaces=surfaces,
    )
    solved_coupling = bool(
        np.asarray(mdo_result.input_conventions.couple_eta_cstar)
    )
    if solved_coupling != bool(couple_eta_cstar):
        raise ValueError(
            "EngineState/coupling mismatch: state was solved with "
            f"couple_eta_cstar={solved_coupling}, but re-evaluation requested "
            f"{bool(couple_eta_cstar)}"
        )


def _authoritative_scalars(result: Any) -> dict[str, float]:
    """Extract the actual EnginePerformance dataclass plus contour scalars."""

    out: dict[str, float] = {}
    perf = result.performance
    aliases = {
        "Isp": "Isp",
        "mdot": "m_dot",
        "thrust": "thrust",
        "Cf": "Cf_actual",
        "Cf_ideal": "Cf_ideal",
        "c_star": "c_star",
        "c_star_delivered": "c_star_effective",
        "Pe": "Pe",
        "Me": "Me",
        "Pc": "Pc",
    }
    for public, attr in aliases.items():
        value = getattr(perf, attr, None)
        if isinstance(value, (int, float, np.number)):
            out[public] = float(value)
    contour = result.contour
    if contour is not None:
        if contour.get("Rt") is not None:
            out["Rt"] = float(contour["Rt"])
        if contour.get("epsilon") is not None:
            out["eps"] = float(contour["epsilon"])
    return out


@dataclass(frozen=True)
class ReEvaluation:
    """MDO screening result, authoritative result, and versioned comparison."""

    mdo: dict[str, Any]
    authoritative: dict[str, Any]
    deltas: dict[str, float]
    result: Any
    warnings: tuple[str, ...]
    electric_pump_result: Any | None = None
    mdo_snapshot: EngineAnalysisSnapshot | None = None
    authoritative_snapshot: EngineAnalysisSnapshot | None = None
    comparison: SnapshotComparison | None = None
    pump_sizing_error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def reevaluate(
    design: Any,
    mission: MissionSpec,
    *,
    mdo_summary: Mapping[str, Any],
    mdo_result: Any | None = None,
    optimizer_metadata: Mapping[str, Any] | None = None,
    couple_eta_cstar: bool = False,
    size_pumps: bool = True,
    pump_spec: Any | None = None,
    mdo_surfaces: Any | None = None,
    **kw: Any,
) -> ReEvaluation:
    """Run an MDO optimum through the authoritative workflow and compare it.

    ``mdo_result`` should be supplied by callers that already have the solved
    engine state.  For backward-compatible call sites that only pass a summary,
    the function attempts one reporting solve; a failure does not discard the
    authoritative result or scalar discrepancy report.  When ``mdo_result`` was
    solved with caller-supplied chamber-property surfaces, pass that exact
    object as ``mdo_surfaces=`` so its evaluator fingerprint can be validated.
    """

    from raosim.design import design_nozzle_v2

    d = _as_design_dict(design)
    mdo_solve_error = None
    if mdo_result is None:
        try:
            mdo_result = _solve_mdo_for_snapshot(
                d,
                mission,
                couple_eta_cstar=couple_eta_cstar,
                surfaces=mdo_surfaces,
            )
        except Exception as exc:
            # The scalar summary still permits a useful authoritative
            # discrepancy report.  The exception is recorded below.
            mdo_result = None
            mdo_solve_error = f"{type(exc).__name__}: {exc}"

    if mdo_result is not None:
        solved_of = _mdo_effective_of(mdo_result)
        result_layout_is_variable = bool(
            np.asarray(mdo_result.input_conventions.of_is_variable)
        ) if (
            hasattr(mdo_result, "input_conventions")
            and hasattr(mdo_result.input_conventions, "of_is_variable")
        ) else mission.design_layout().of_is_variable
        # A fixed-layout caller may still own a legacy dict whose OF slot was
        # an intent sentinel.  Normalize that non-authoritative slot from the
        # solved result before validating the fixed physical v2 contract.  A
        # variable-layout value remains caller-owned and mismatch-checked.
        if not result_layout_is_variable:
            d["OF"] = solved_of
        _validate_mdo_state_handoff(
            mdo_result,
            d,
            mission,
            couple_eta_cstar=couple_eta_cstar,
            surfaces=mdo_surfaces,
        )
        kw.setdefault("mdot_total", _mdo_total_mass_flow(mdo_result))
        kw.setdefault("eta_cstar", _mdo_eta_cstar(mdo_result))
        kw.setdefault("eta_CF", float(mission.eta_CF))
        kw.setdefault("effective_of", solved_of)
        if "thermo_mode" not in kw and "pinned_chamber_state" not in kw:
            pinned_chamber_state = _pinned_chamber_state_from_mdo(
                mdo_result,
                mission,
                surfaces=mdo_surfaces,
            )
            if pinned_chamber_state is not None:
                kw["pinned_chamber_state"] = pinned_chamber_state
    design_input = to_design_input(d, mission, **kw)

    # Remediation item 9: the resolved contract is the authority on every
    # convention the two pipelines share.  Building it here and crosschecking
    # the traditional handoff turns an input divergence into an immediate,
    # named warning instead of an output discrepancy someone has to diagnose
    # backwards later.  The contract is the input authority; ``to_design_input``
    # remains the builder until it is switched over to consume it.
    resolved_inputs = None
    resolved_inputs_drift: tuple[str, ...] = ()
    resolved_inputs_error: str | None = None
    if mdo_result is not None:
        from raosim.mdo.resolved_inputs import (
            crosscheck_design_input,
            resolve_engine_inputs,
        )

        try:
            resolved_inputs = resolve_engine_inputs(
                d,
                mission,
                effective_of=float(kw["effective_of"]),
                of_source=(
                    "optimized" if result_layout_is_variable
                    else "mission_nominal"
                ),
                total_mass_flow=float(kw["mdot_total"]),
                eta_cstar_effective=float(kw["eta_cstar"]),
                contour_method=str(kw.get("contour_method", "bezier")),
                surfaces=mdo_surfaces,
            )
            resolved_inputs_drift = crosscheck_design_input(
                resolved_inputs, design_input
            )
        except Exception as exc:
            resolved_inputs_error = f"{type(exc).__name__}: {exc}"

    result = design_nozzle_v2(design_input)

    pump_result = None
    pump_error = None
    if size_pumps:
        ledger = _feed_ledger_from_result(result)
        if _usable_feed_ledger(ledger):
            try:
                from raosim.pumps import size_electric_pumps

                spec = pump_spec or _pump_spec(d, mission, mdo_result)
                pump_result = size_electric_pumps(ledger, spec)
            except Exception as exc:
                pump_error = f"{type(exc).__name__}: {exc}"
        else:
            pump_error = (
                "authoritative injector report did not produce a usable "
                "two-stream feed ledger with pump pressure rise/head"
            )

    auth = _authoritative_scalars(result)
    mdo = dict(mdo_summary)
    deltas = {
        key: float(auth[key]) - float(mdo[key])
        for key in auth.keys() & mdo.keys()
        if isinstance(auth[key], (int, float, np.number))
        and isinstance(mdo[key], (int, float, np.number))
    }
    warnings = list(result.warnings or ())
    if resolved_inputs_drift:
        warnings.append(
            "The traditional handoff disagrees with the resolved engine-input "
            "contract on "
            f"{len(resolved_inputs_drift)} scalar(s), so the two pipelines are "
            "not analyzing the same engine: "
            + "; ".join(resolved_inputs_drift)
        )
    if resolved_inputs_error:
        warnings.append(
            "Resolved engine-input contract unavailable, so the traditional "
            "handoff could not be crosschecked against it: "
            + resolved_inputs_error
        )
    solved_coupling = bool(
        np.asarray(mdo_result.input_conventions.couple_eta_cstar)
    ) if (
        mdo_result is not None
        and hasattr(mdo_result, "input_conventions")
        and hasattr(mdo_result.input_conventions, "couple_eta_cstar")
    ) else bool(couple_eta_cstar)
    snapshot_optimizer_metadata = dict(optimizer_metadata or {})
    snapshot_optimizer_metadata.setdefault(
        "eta_cstar_handoff",
        "frozen_from_solved_mdo_state",
    )
    snapshot_optimizer_metadata.setdefault(
        "thermochemistry_handoff",
        (
            "pinned_from_solved_mdo_state"
            if getattr(design_input.thermo, "pinned_chamber_state", None)
            is not None
            else f"host_provider:{design_input.thermo.mode}"
        ),
    )
    snapshot_optimizer_metadata.setdefault(
        "mdo_spray_cstar_coupling_enabled",
        solved_coupling,
    )
    snapshot_optimizer_metadata.setdefault(
        "traditional_spray_cstar_coupling_rerun",
        False,
    )
    if resolved_inputs is not None:
        # Content-addressed identity of the exact inputs both pipelines ran,
        # so a snapshot can be replayed and a parity claim can be checked
        # against the inputs rather than assumed.
        snapshot_optimizer_metadata.setdefault(
            "resolved_engine_inputs",
            {
                "schema_version": resolved_inputs.schema_version,
                "digest": resolved_inputs.digest(),
                "mixture_ratio": resolved_inputs.propellant.mixture_ratio,
                "mixture_ratio_source":
                    resolved_inputs.propellant.mixture_ratio_source,
                "liner_material": resolved_inputs.material.liner_name,
                "closeout_material": resolved_inputs.material.closeout_name,
                "model_identities": dict(resolved_inputs.model_identities),
                "unavailable": dict(resolved_inputs.unavailable),
                "traditional_handoff_crosscheck": (
                    "agrees" if not resolved_inputs_drift else "disagrees"
                ),
            },
        )
    if solved_coupling:
        warnings.append(
            "The traditional re-evaluation freezes the effective eta_cstar "
            "resolved by the coupled MDO state; it does not independently "
            "rerun the MDO spray-to-cstar screening correlation."
        )
    if mdo_solve_error:
        warnings.append(
            "MDO EngineState unavailable for full snapshot comparison: "
            + mdo_solve_error
        )
    if pump_error:
        warnings.append(f"Electric-pump authoritative sizing unavailable: {pump_error}")

    # Register the contract report before adapting the result so the
    # authoritative snapshot's artifact table contains its own stable handoff.
    snapshot_report_path = _snapshot_report_path(design_input)
    if snapshot_report_path is not None:
        result.files["engine_analysis_snapshot"] = snapshot_report_path

    mdo_snapshot = None
    authoritative_snapshot = snapshot_from_traditional(
        result,
        pump_result,
        optimizer_metadata=snapshot_optimizer_metadata,
    )
    authoritative_snapshot = replace(
        authoritative_snapshot,
        warnings=tuple(
            dict.fromkeys(
                (*authoritative_snapshot.warnings, *(str(w) for w in warnings))
            )
        ),
    )
    comparison = None
    if mdo_result is not None:
        mdo_snapshot = snapshot_from_mdo(
            mdo_result,
            d,
            mission,
            optimizer_metadata=snapshot_optimizer_metadata,
            surfaces=mdo_surfaces,
        )
        comparison = compare_snapshots(mdo_snapshot, authoritative_snapshot)

    if snapshot_report_path is not None:
        # The report hash is embedded in downstream JSON artifacts.  Attachment
        # failures must also live in the authoritative snapshot, so converge the
        # report/warning set before accepting the final hash.  In the normal
        # success case this is one pass; a deterministic failure takes two.
        attachment_warnings: set[str] = set()
        for _ in range(5):
            _write_snapshot_report(
                snapshot_report_path,
                authoritative=authoritative_snapshot,
                mdo=mdo_snapshot,
                comparison=comparison,
            )
            observed = tuple(
                str(item)
                for item in _attach_snapshot_handoff(
                    result,
                    snapshot_report_path,
                    authoritative_snapshot,
                )
            )
            new_warnings = [
                item for item in observed
                if item not in attachment_warnings
            ]
            if not new_warnings:
                break
            attachment_warnings.update(new_warnings)
            warnings.extend(new_warnings)
            authoritative_snapshot = replace(
                authoritative_snapshot,
                warnings=tuple(
                    dict.fromkeys(
                        (
                            *authoritative_snapshot.warnings,
                            *new_warnings,
                        )
                    )
                ),
            )
        else:
            stabilization_warning = (
                "authoritative snapshot artifact attachments did not reach a "
                "stable warning set after five passes"
            )
            warnings.append(stabilization_warning)
            authoritative_snapshot = replace(
                authoritative_snapshot,
                warnings=tuple(
                    dict.fromkeys(
                        (
                            *authoritative_snapshot.warnings,
                            stabilization_warning,
                        )
                    )
                ),
            )
            _write_snapshot_report(
                snapshot_report_path,
                authoritative=authoritative_snapshot,
                mdo=mdo_snapshot,
                comparison=comparison,
            )
            _attach_snapshot_handoff(
                result,
                snapshot_report_path,
                authoritative_snapshot,
            )

    return ReEvaluation(
        mdo=mdo,
        authoritative=auth,
        deltas=deltas,
        result=result,
        warnings=tuple(dict.fromkeys(str(w) for w in warnings)),
        electric_pump_result=pump_result,
        mdo_snapshot=mdo_snapshot,
        authoritative_snapshot=authoritative_snapshot,
        comparison=comparison,
        pump_sizing_error=pump_error,
        metadata={
            "contract_version": authoritative_snapshot.contract_version,
            "optimizer_metadata": snapshot_optimizer_metadata,
            "design_input": design_input,
            "mdo_snapshot_error": mdo_solve_error,
            "authoritative_snapshot_report": (
                str(snapshot_report_path)
                if snapshot_report_path is not None
                else None
            ),
        },
    )


def summarise(reev: ReEvaluation) -> str:
    """Human-readable scalar discrepancy and snapshot-comparison report."""

    lines = ["  MDO screening vs authoritative pipeline:"]
    if not reev.deltas:
        lines.append("    (no directly comparable scalars were produced)")
    for key, delta in sorted(reev.deltas.items()):
        base = float(reev.mdo.get(key, float("nan")))
        pct = (100.0 * delta / base) if base else float("nan")
        flag = "  <-- check" if abs(pct) > 5.0 else ""
        lines.append(
            f"    {key:<16s} mdo={base:12.4g}  "
            f"auth={reev.authoritative[key]:12.4g}  "
            f"Δ={delta:+.4g} ({pct:+.1f}%){flag}"
        )
    if reev.comparison is not None:
        different = sum(
            item.status == "different"
            for item in (
                *reev.comparison.scalars.values(),
                *reev.comparison.profiles.values(),
            )
        )
        not_comparable = sum(
            item.not_comparable
            for item in (
                *reev.comparison.scalars.values(),
                *reev.comparison.profiles.values(),
            )
        )
        lines.append(
            "    snapshot contract "
            f"{reev.authoritative_snapshot.contract_version}: "
            f"{reev.comparison.comparable_count} comparable, "
            f"{different} outside tolerance, "
            f"{not_comparable} explicitly unavailable"
        )
    if reev.electric_pump_result is not None:
        lines.append(
            "    authoritative electric-pump sizing: "
            + ("feasible" if reev.electric_pump_result.feasible else "gates failed")
        )
    if reev.warnings:
        lines.append("  pipeline warnings:")
        for warning in reev.warnings[:8]:
            lines.append(f"    - {warning}")
    return "\n".join(lines)


__all__ = [
    "ReEvaluation",
    "reevaluate",
    "summarise",
    "to_design_input",
]
