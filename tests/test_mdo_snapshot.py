"""End-to-end tests for the versioned MDO/traditional output bridge."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from raosim.cea import THERMO_PINNED_CHAMBER
from raosim.mdo.engine import solve_engine
from raosim.mdo.postprocess import (
    _attach_snapshot_handoff,
    reevaluate,
    to_design_input,
)
from raosim.mdo.schema import DesignVector, MissionSpec
from raosim.mdo.snapshot import (
    CONTRACT_VERSION,
    available,
    compare_snapshots,
    maybe,
    snapshot_from_mdo,
)
from raosim.mdo.properties import constant_chamber_surfaces
from raosim.mdo.state import MASS_FIELD_NAMES, solve_engine_state


@pytest.fixture(scope="module")
def mission() -> MissionSpec:
    return MissionSpec()


@pytest.fixture(scope="module")
def design_vector() -> DesignVector:
    return DesignVector(
        Pc=3.0e6,
        eps=8.0,
        dp_f_frac=0.23,
        dp_o_frac=0.27,
        D_pintle=0.020,
        N_rpm=30_000.0,
        channel_width=5.0e-4,
        channel_height=1.5e-3,
        film_frac=0.10,
        t_wall=8.0e-4,
    )


@pytest.fixture(scope="module")
def engine_result(mission, design_vector):
    return solve_engine(design_vector, mission)


@pytest.fixture(scope="module")
def engine_state(mission, design_vector):
    return solve_engine_state(design_vector, mission)


def test_to_design_input_maps_shared_conventions(mission, design_vector):
    d = {key: float(value) for key, value in design_vector.as_dict().items()}
    mapped = to_design_input(d, mission)

    assert mapped.thermo.propellant_name == mission.propellant_name
    assert mapped.thermo.mixture_ratio == pytest.approx(mission.OF)
    eta_cstar = mission.eta_cstar * (
        1.0 - mission.film_cstar_penalty * d["film_frac"]
    )
    assert mapped.thermo.eta_cstar == pytest.approx(eta_cstar)
    assert mapped.thermo.eta_CF == pytest.approx(mission.eta_CF)
    assert mapped.thermo.eta_Isp == pytest.approx(
        eta_cstar * mission.eta_CF
    )
    assert mapped.ambient.Pa == pytest.approx(mission.Pa)
    assert mapped.cooling.coolant == "RP-1"
    assert mapped.cooling.coolant_density == pytest.approx(mission.rho_cool)
    assert mapped.cooling.coolant_cp == pytest.approx(mission.cp_cool)
    assert mapped.cooling.coolant_conductivity == pytest.approx(mission.k_cool)
    assert mapped.cooling.channel_count == mission.n_channels
    assert mapped.cooling.fuel_film_mass_flow > 0.0
    assert mapped.cooling.coolant_mass_flow + mapped.cooling.fuel_film_mass_flow > 0
    assert mapped.cooling.coolant_outlet_pressure == pytest.approx(
        d["Pc"] * (1.0 + d["dp_f_frac"])
    )
    assert mapped.injector.type == "pintle"
    assert mapped.injector.architecture == "fixed_discrete"
    assert mapped.injector.sizing == "auto"
    assert mapped.injector.fuel_dp_fraction == pytest.approx(d["dp_f_frac"])
    assert mapped.injector.oxidizer_dp_fraction == pytest.approx(d["dp_o_frac"])
    assert mapped.injector.fuel_cd == pytest.approx(mission.injector_cd_fuel)
    assert mapped.injector.oxidizer_cd == pytest.approx(mission.injector_cd_ox)
    assert mapped.injector.fuel.density == pytest.approx(mission.rho_fuel)
    assert mapped.injector.oxidizer.density == pytest.approx(mission.rho_ox)
    assert mapped.injector.fuel.vapor_pressure == pytest.approx(
        mission.p_vapor_fuel
    )
    assert mapped.injector.oxidizer.vapor_pressure == pytest.approx(
        mission.p_vapor_ox
    )
    assert mapped.injector.feed_system.fuel.tank_pressure == pytest.approx(
        mission.P_tank_fuel
    )
    assert mapped.injector.feed_system.oxidizer.tank_pressure == pytest.approx(
        mission.P_tank_ox
    )
    assert mapped.throat_geometry.upstream_radius_ratio == pytest.approx(
        mission.throat_ru_factor
    )
    assert mapped.throat_geometry.downstream_radius_ratio == pytest.approx(
        mission.throat_rd_factor
    )
    assert mapped.material.conductivity == pytest.approx(mission.k_wall)
    assert mapped.material.elastic_modulus == pytest.approx(mission.liner_E)
    assert mapped.material.thermal_expansion == pytest.approx(
        mission.liner_alpha
    )
    assert mapped.material.structural_fos == pytest.approx(
        mission.liner_structural_fos
    )
    assert mapped.material.yield_strength == pytest.approx(
        mission.liner_sigma_allow * mission.liner_structural_fos
    )

    selected = to_design_input(
        d,
        mission,
        thermo_mode="cea_frozen",
        contour_method="rao_variational_moc",
    )
    assert selected.thermo.mode == "cea_frozen"
    assert selected.method == "rao_variational_moc"

    configured = to_design_input(
        d,
        mission,
        contour_method="rao_variational_moc",
        host_rao_solver_options={
            "n_control": 18,
            "n_kernel": 20,
            "max_nfev": 321,
            "solver_backend": "numpy",
            "wall_method": "bde",
            "kernel_d_fraction_max": 0.7,
            "physics_weight": 1.0,
        },
    )
    assert configured.host_rao_solver.n_control == 18
    assert configured.host_rao_solver.n_kernel == 20
    assert configured.host_rao_solver.max_nfev == 321
    assert configured.host_rao_solver.solver_backend == "numpy"
    assert configured.host_rao_solver.wall_method == "bde"
    assert configured.host_rao_solver.kernel_d_fraction_max == pytest.approx(
        0.7
    )
    assert configured.host_rao_solver.physics_weight == pytest.approx(1.0)

    cea_backed = to_design_input(
        d,
        replace(mission, cea_table_path="manufactured_surface_fixture.npz"),
    )
    assert cea_backed.thermo.mode == "cea_frozen"


def test_engine_state_is_primary_snapshot_path(mission, design_vector):
    state = solve_engine_state(design_vector, mission)
    snap = snapshot_from_mdo(
        state,
        mission=mission,
        optimizer_metadata={"method": "SLSQP", "iterations": 12},
    )

    assert snap.contract_version == CONTRACT_VERSION
    assert snap.source == "mdo"
    assert snap.source_result is state
    assert snap.performance["specific_impulse_ideal_s"].available
    assert snap.performance["specific_impulse_delivered_s"].value == pytest.approx(
        float(state.performance.Isp_delivered)
    )
    assert snap.performance["cf_ideal"].value == pytest.approx(
        float(state.performance.Cf_ideal)
    )
    assert snap.performance["cf_delivered"].value == pytest.approx(
        float(state.performance.Cf_delivered)
    )
    assert snap.cooling["coolant_mass_flow_kg_s"].value == pytest.approx(
        float(state.performance.mdot_regen_jacket)
    )
    # Thrust-chamber structure is integrated on the station grid, so it is a
    # real mass with a real load-path split.
    assert snap.masses["thrust_chamber_mass_kg"].value == pytest.approx(
        float(state.masses.values[MASS_FIELD_NAMES.index("thrust_chamber_mass")])
    )
    assert snap.masses["thrust_chamber_mass_kg"].value > 0.0
    assert snap.masses["thrust_chamber_mass_kg"].availability_reason is None
    assert snap.masses["thrust_chamber_mass_kg"].value == pytest.approx(
        snap.masses["thrust_chamber_liner_mass_kg"].value
        + snap.masses["thrust_chamber_land_mass_kg"].value
        + snap.masses["thrust_chamber_closeout_mass_kg"].value
    )
    # Injector hardware and a total dry mass remain honestly unavailable.
    assert snap.masses["injector_mass_kg"].value is None
    assert snap.masses["injector_mass_kg"].availability_reason
    assert snap.masses["total_engine_package_mass_kg"].value is None
    assert snap.masses["total_engine_package_mass_kg"].availability_reason
    assert snap.optimizer_metadata["method"] == "SLSQP"
    assert snap.provenance["input_conventions"].available
    conventions = snap.provenance["input_conventions"].value
    assert conventions["eta_cstar_nominal"] == pytest.approx(
        mission.eta_cstar
    )
    assert conventions["eta_cstar_effective"] == pytest.approx(
        float(state.performance.eta_cstar)
    )
    json.dumps(snap.to_dict(), allow_nan=False)

    same = compare_snapshots(snap, snap)
    assert same.profiles["geometry.radius_profile_m"].status == "within_tolerance"
    assert same.scalars[
        "performance.specific_impulse_delivered_s"
    ].status == "within_tolerance"
    # The chamber mass is now a real integrated quantity, so it compares.
    assert not same.scalars["masses.thrust_chamber_mass_kg"].not_comparable
    assert same.scalars[
        "masses.thrust_chamber_mass_kg"
    ].status == "within_tolerance"
    # Injector hardware mass is still unavailable on the MDO side, so it must
    # remain explicitly not-comparable rather than comparing zero to zero.
    assert same.scalars["masses.injector_mass_kg"].not_comparable

    with pytest.raises(ValueError, match="convention mismatch"):
        snapshot_from_mdo(
            state,
            mission=replace(mission, P_tank_fuel=mission.P_tank_fuel + 1.0),
        )
    with pytest.raises(ValueError, match="propellant_name"):
        snapshot_from_mdo(
            state,
            mission=replace(mission, propellant_name="LOX/LH2"),
        )
    with pytest.raises(ValueError, match="fingerprint"):
        snapshot_from_mdo(
            state,
            mission=replace(
                mission,
                film_decay_exponent=mission.film_decay_exponent + 0.01,
            ),
        )
    with pytest.raises(ValueError, match="design mismatch"):
        snapshot_from_mdo(
            state,
            replace(design_vector, eps=design_vector.eps + 0.01),
            mission,
        )
    with pytest.raises(ValueError, match="unsupported EngineState schema version"):
        snapshot_from_mdo(
            state._replace(schema_version=np.asarray(2)),
            mission=mission,
        )


def test_custom_surface_state_requires_and_accepts_exact_surface_identity(
    mission, design_vector
):
    surfaces = constant_chamber_surfaces(
        gamma=mission.gamma + 0.005,
        Tc=mission.Tc + 10.0,
        R_gas=mission.R_gas + 1.0,
    )
    state = solve_engine_state(design_vector, mission, surfaces=surfaces)

    with pytest.raises(ValueError, match="surface signature differs"):
        snapshot_from_mdo(state, mission=mission)

    snap = snapshot_from_mdo(
        state,
        mission=mission,
        surfaces=surfaces,
    )
    thermo = snap.provenance["thermochemistry"].value
    assert thermo["surface_provenance"] == surfaces.provenance
    assert len(thermo["surface_fingerprint"]) == 8


def test_custom_surface_state_is_pinned_into_the_host_reevaluation(
    mission, design_vector
):
    surfaces = constant_chamber_surfaces(
        gamma=mission.gamma + 0.01,
        Tc=mission.Tc + 75.0,
        R_gas=mission.R_gas + 7.0,
    )
    state = solve_engine_state(design_vector, mission, surfaces=surfaces)
    design = {
        key: float(value) for key, value in design_vector.as_dict().items()
    }

    reev = reevaluate(
        design,
        mission,
        mdo_result=state,
        mdo_surfaces=surfaces,
        mdo_summary={"Isp": float(state.performance.Isp_delivered)},
        size_pumps=False,
    )

    assert reev.result.thermochemistry.mode == THERMO_PINNED_CHAMBER
    assert reev.result.propellant.gamma == pytest.approx(
        float(state.performance.gamma)
    )
    assert reev.result.propellant.Tc == pytest.approx(
        float(state.performance.Tc)
    )
    assert reev.result.propellant.R_gas == pytest.approx(
        float(state.performance.R_gas)
    )
    assert reev.result.propellant.c_star == pytest.approx(
        float(state.performance.cstar_ideal)
    )
    for path in (
        "performance.c_star_ideal_m_s",
        "performance.c_star_delivered_m_s",
        "performance.eta_cstar",
        "performance.eta_cf",
    ):
        assert reev.comparison.scalars[path].status == "within_tolerance"


def test_nested_nonfinite_available_values_are_rejected_or_marked_unavailable():
    payload = {"outer": {"finite": 1.0, "bad": [2.0, np.nan]}}
    with pytest.raises(ValueError, match=r"\$\.outer\.bad\[1\]"):
        available(payload)
    field = maybe(payload, "nested contract payload is invalid")
    assert not field.available
    assert "$.outer.bad[1]" in field.availability_reason


def test_legacy_engine_result_snapshot_remains_supported(
    mission, design_vector, engine_result
):
    snap = snapshot_from_mdo(engine_result, design_vector, mission)
    assert snap.performance["mass_flow_total_kg_s"].value == pytest.approx(
        float(engine_result.mdot)
    )
    assert snap.injector["fuel_dp_fraction"].value == pytest.approx(
        float(design_vector.dp_f_frac)
    )
    assert snap.geometry["area_ratio_profile"].available
    # The legacy EngineResult path reads the chamber mass out of the same
    # ledger the state path uses, so it is available here too; only the
    # injector branch stays unavailable.
    assert snap.masses["thrust_chamber_mass_kg"].value > 0.0
    assert snap.masses["injector_mass_kg"].value is None
    assert snap.masses["injector_mass_kg"].availability_reason


def test_authoritative_handoff_updates_cad_json_sidecar(
    mission, engine_state, tmp_path
):
    snapshot = snapshot_from_mdo(engine_state, mission=mission)
    snapshot_path = tmp_path / "engine_analysis_snapshot_v1.json"
    snapshot_path.write_text(
        json.dumps(snapshot.to_dict(), allow_nan=False),
        encoding="utf-8",
    )
    cad_metadata = tmp_path / "thrust_chamber_wall.cad.json"
    cad_metadata.write_text(
        json.dumps({"geometry": "test-sidecar"}),
        encoding="utf-8",
    )
    pintle_parameters = tmp_path / "pintle_parameters.json"
    pintle_parameters.write_text(
        json.dumps({"injector": "test-parameters"}),
        encoding="utf-8",
    )
    result = SimpleNamespace(
        files={
            "step_cad_metadata": cad_metadata,
            # This is the real key shape that the former suffix matcher missed.
            "pintle_parameters_json": pintle_parameters,
        }
    )

    assert not _attach_snapshot_handoff(
        result, snapshot_path, snapshot
    )
    for artifact in (cad_metadata, pintle_parameters):
        handoff = json.loads(artifact.read_text(encoding="utf-8"))[
            "authoritative_analysis_snapshot"
        ]
        assert handoff["snapshot_path"] == str(snapshot_path)
        assert handoff["snapshot_sha256"] == hashlib.sha256(
            snapshot_path.read_bytes()
        ).hexdigest()
        assert handoff["optimizer_metadata"] == {}


def test_attachment_failures_are_persisted_in_the_authoritative_report(
    mission, design_vector, engine_state, tmp_path, monkeypatch
):
    warning = "manufactured authoritative snapshot attachment failure"
    calls: list[int] = []

    def failing_attachment(*_args, **_kwargs):
        calls.append(1)
        return (warning,)

    monkeypatch.setattr(
        "raosim.mdo.postprocess._attach_snapshot_handoff",
        failing_attachment,
    )
    design = {
        key: float(value) for key, value in design_vector.as_dict().items()
    }
    reev = reevaluate(
        design,
        mission,
        mdo_result=engine_state,
        mdo_summary={"Isp": float(engine_state.performance.Isp_delivered)},
        size_pumps=False,
        output_dir=tmp_path,
    )

    # First pass discovers the failure; second pass writes the warning-bearing
    # snapshot and establishes the stable final report hash.
    assert len(calls) == 2
    assert warning in reev.warnings
    assert warning in reev.authoritative_snapshot.warnings
    report = json.loads(
        (tmp_path / "engine_analysis_snapshot_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert warning in report["authoritative"]["warnings"]


def test_reevaluation_rejects_design_and_coupling_mismatch(
    mission, design_vector, engine_state
):
    design = {
        key: float(value) for key, value in design_vector.as_dict().items()
    }
    with pytest.raises(ValueError, match="design mismatch"):
        reevaluate(
            {**design, "eps": design["eps"] + 0.01},
            mission,
            mdo_result=engine_state,
            mdo_summary={},
            size_pumps=False,
        )

    coupled_marker = engine_state._replace(
        input_conventions=engine_state.input_conventions._replace(
            couple_eta_cstar=np.asarray(True)
        )
    )
    with pytest.raises(ValueError, match="coupling mismatch"):
        reevaluate(
            design,
            mission,
            mdo_result=coupled_marker,
            mdo_summary={},
            size_pumps=False,
        )


def test_reevaluation_validates_mission_before_creating_artifacts(
    mission, design_vector, engine_state, tmp_path
):
    design = {
        key: float(value) for key, value in design_vector.as_dict().items()
    }
    output_dir = tmp_path / "must_not_be_created"

    with pytest.raises(ValueError, match="convention mismatch"):
        reevaluate(
            design,
            replace(
                mission,
                P_tank_fuel=mission.P_tank_fuel + 1.0,
            ),
            mdo_result=engine_state,
            mdo_summary={},
            output_dir=output_dir,
        )

    assert not output_dir.exists()


def test_authoritative_reevaluation_is_end_to_end_and_sizes_pumps(
    mission, design_vector, engine_state, tmp_path
):
    design = {
        key: float(value) for key, value in design_vector.as_dict().items()
    }
    reev = reevaluate(
        design,
        mission,
        mdo_result=engine_state,
        mdo_summary={
            "Isp": float(engine_state.performance.Isp_delivered),
            "Rt": float(engine_state.performance.Rt),
            "eps": float(design_vector.eps),
            "mdot": float(engine_state.performance.mdot_total),
            "thrust": float(mission.thrust),
        },
        optimizer_metadata={"success": True, "method": "test"},
        output_dir=tmp_path,
    )

    # Regression for the old ``isinstance(performance, dict)`` bug.
    assert reev.authoritative["Isp"] == pytest.approx(
        reev.result.performance.Isp
    )
    assert reev.authoritative["mdot"] == pytest.approx(
        reev.result.performance.m_dot
    )
    assert reev.authoritative["thrust"] == pytest.approx(
        reev.result.performance.thrust
    )
    assert {"Isp", "Rt", "eps", "mdot", "thrust"} <= reev.deltas.keys()

    mapped = reev.result.input
    assert mapped.thermo.mode == THERMO_PINNED_CHAMBER
    pinned = mapped.thermo.pinned_chamber_state
    assert pinned is not None
    assert pinned.gamma == pytest.approx(float(engine_state.performance.gamma))
    assert pinned.Tc == pytest.approx(float(engine_state.performance.Tc))
    assert pinned.R_gas == pytest.approx(float(engine_state.performance.R_gas))
    assert pinned.c_star_ideal == pytest.approx(
        float(engine_state.performance.cstar_ideal)
    )
    assert len(pinned.surface_fingerprint or "") == 64
    assert reev.result.thermochemistry.source.startswith(
        "pinned_from_mdo_EngineState:"
    )
    assert reev.metadata["optimizer_metadata"][
        "thermochemistry_handoff"
    ] == "pinned_from_solved_mdo_state"
    assert mapped.injector.type == "pintle"
    assert mapped.injector.fuel_dp_fraction == pytest.approx(
        design["dp_f_frac"]
    )
    assert mapped.injector.oxidizer_dp_fraction == pytest.approx(
        design["dp_o_frac"]
    )
    assert mapped.cooling.coolant == "RP-1"
    assert mapped.ambient.Pa == pytest.approx(mission.Pa)
    assert mapped.throat_geometry.downstream_radius_ratio == pytest.approx(
        mission.throat_rd_factor
    )

    assert reev.electric_pump_result is not None
    assert reev.pump_sizing_error is None
    assert set(reev.electric_pump_result.lines) == {"fuel", "oxidizer"}
    assert reev.authoritative_snapshot.authoritative_result is reev.result
    assert (
        reev.authoritative_snapshot.auxiliary_results["electric_pumps"]
        is reev.electric_pump_result
    )
    assert reev.mdo_snapshot is not None
    assert reev.comparison is not None
    assert reev.authoritative_snapshot.constraints_gates[
        "all_constraints_feasible"
    ].available
    assert reev.authoritative_snapshot.constraints_gates[
        "physics_feasible"
    ].available
    assert reev.comparison.comparable_count > 80
    for section_name in (
        "performance",
        "geometry",
        "thermal",
        "cooling",
        "injector",
        "feed_electrical",
        "masses",
        "constraints_gates",
        "provenance",
        "artifacts",
    ):
        assert set(getattr(reev.mdo_snapshot, section_name).fields) == set(
            getattr(reev.authoritative_snapshot, section_name).fields
        )
    # These are true cross-pipeline parity checks, not self-comparisons against
    # a simplified mirror.  The shared performance convention must remain
    # identical after design_nozzle_v2 re-evaluates the design.
    for path in (
        "performance.chamber_pressure_pa",
        "performance.thrust_n",
        "performance.specific_impulse_delivered_s",
        "performance.mass_flow_total_kg_s",
        "performance.mixture_ratio",
        "performance.cf_ideal",
        "performance.cf_delivered",
        "performance.c_star_ideal_m_s",
        "performance.c_star_delivered_m_s",
        "performance.eta_cstar",
        "performance.eta_cf",
        "geometry.throat_radius_m",
        "geometry.expansion_ratio",
        "geometry.throat_upstream_radius_ratio",
        "geometry.throat_downstream_radius_ratio",
        "cooling.film_mass_flow_kg_s",
        "cooling.film_fraction_of_fuel",
        "feed_electrical.fuel_density_kg_m3",
        "feed_electrical.oxidizer_density_kg_m3",
        "feed_electrical.fuel_vapor_pressure_pa",
        "feed_electrical.oxidizer_vapor_pressure_pa",
        "feed_electrical.fuel_npsh_available_pa",
        "feed_electrical.oxidizer_npsh_available_pa",
    ):
        assert reev.comparison.scalars[path].status == "within_tolerance"
    for path in (
        "cooling.method",
        "cooling.coolant_name",
        "cooling.fuel_flow_topology",
        "injector.type",
        "injector.architecture",
        "feed_electrical.architecture",
    ):
        assert reev.comparison.scalars[path].status == "within_tolerance"
    traditional_conventions = (
        reev.authoritative_snapshot.provenance["input_conventions"].value
    )
    assert traditional_conventions["eta_cstar_effective"] == pytest.approx(
        reev.result.performance.eta_cstar
    )
    assert traditional_conventions["eta_cstar_nominal"]["value"] is None
    assert traditional_conventions["eta_cstar_nominal"][
        "availability_reason"
    ]
    assert traditional_conventions["cooling_fraction"] == pytest.approx(
        mission.cooling_fraction
    )
    assert traditional_conventions["fuel_film_fraction"] == pytest.approx(
        design["film_frac"]
    )
    material_metadata = (
        reev.authoritative_snapshot.provenance[
            "material_assumptions"
        ].value
    )
    assert material_metadata["max_heat_flux"]["value"] is None
    assert "positive infinity" in material_metadata["max_heat_flux"][
        "availability_reason"
    ]
    assert np.isfinite(
        reev.comparison.scalars[
            "performance.specific_impulse_delivered_s"
        ].absolute_delta
    )
    # These are measured cross-model discrepancy ceilings, not declarations
    # that the analytic MDO mirrors are authoritative-equivalent.
    scalar_ceilings = {
        "cooling.coolant_pressure_drop_pa": 0.07,
        "feed_electrical.electric_power_total_w": 0.02,
        "feed_electrical.fuel_required_pressure_rise_pa": 0.03,
    }
    for path, ceiling in scalar_ceilings.items():
        assert reev.comparison.scalars[path].relative_delta <= ceiling
    profile_ceilings = {
        "geometry.mach_profile": 0.03,
        "geometry.radius_profile_m": 0.25,
        "geometry.area_ratio_profile": 0.40,
    }
    for path, ceiling in profile_ceilings.items():
        assert reev.comparison.profiles[path].relative_delta <= ceiling
    for path in (
        "thermal.gas_side_wall_temperature_profile",
        "thermal.coolant_side_wall_temperature_profile",
        "thermal.heat_flux_profile",
        "thermal.thermal_stress_profile",
        "thermal.combined_stress_profile",
        "cooling.coolant_temperature_profile",
    ):
        comparison = reev.comparison.profiles[path]
        assert comparison.not_comparable
        assert "does not apply a wall-film" in (
            comparison.not_comparable_reason or ""
        )
    for field_name in (
        "momentum_ratio",
        "spray_half_angle_deg",
        "blockage_factor",
        "transition_margin_m2",
        "fuel_velocity_m_s",
        "fuel_flow_area_m2",
        "slot_width_m",
        "tip_opening_m",
        "tip_branch_area_m2",
        "center_gap_area_m2",
    ):
        assert not reev.mdo_snapshot.injector[field_name].available
        assert not reev.authoritative_snapshot.injector[field_name].available
        assert reev.comparison.scalars[
            f"injector.{field_name}"
        ].not_comparable
    assert reev.comparison.scalars[
        "cooling.coolant_outlet_temperature_k"
    ].not_comparable
    pressure_comparison = reev.comparison.profiles[
        "thermal.pressure_stress_profile"
    ]
    assert not pressure_comparison.not_comparable
    assert np.isfinite(pressure_comparison.relative_delta)
    # The core pump BOM (hydraulic + mechanical + pressure boundary +
    # inlet/outlet ports) is now fully massed, so a real pump and electric
    # package mass exists on the authoritative side.
    pump_mass = reev.authoritative_snapshot.masses["pump_mass_kg"]
    assert pump_mass.available and pump_mass.value > 0.0
    package_mass = reev.authoritative_snapshot.masses[
        "electric_package_mass_kg"
    ]
    assert package_mass.available and package_mass.value >= pump_mass.value
    # Instrumentation mass is still unknown and is deliberately excluded from
    # the core rollup rather than counted as zero.
    bom = reev.authoritative_snapshot.masses["raw_mass_ledger"].value
    assert any(
        row["subsystem"] == "instrumentation"
        and row["mass_estimate_kg"] is None
        for row in bom
    )
    # The smooth-objective branches remain MDO-only concepts.
    assert not reev.authoritative_snapshot.masses[
        "battery_objective_mass_kg"
    ].available
    assert not reev.authoritative_snapshot.masses[
        "electric_package_objective_mass_kg"
    ].available
    split = reev.result.report_sections["injector"]["feed_system"][
        "fuel_flow_split"
    ]
    assert split["status"] == "pass"
    assert split["film_bypass_mass_flow_kg_s"] > 0.0
    regen_gate = next(
        gate
        for gate in reev.result.report_sections["injector"]["gates"]
        if gate["name"] == "regen_fuel_flow_closure"
    )
    assert regen_gate["status"] == "pass"

    report_path = tmp_path / "engine_analysis_snapshot_v1.json"
    assert report_path.is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["authoritative"]["source"] == "traditional"
    assert report["authoritative"]["optimizer_metadata"]["method"] == "test"
    assert report["mdo_screening"]["source"] == "mdo"
    assert report["comparison"]["comparable_count"] > 20
    assert (
        reev.authoritative_snapshot.artifacts["files"].value[
            "engine_analysis_snapshot"
        ]
        == str(report_path)
    )
    traditional_report = json.loads(
        reev.result.files["design_report"].read_text(encoding="utf-8")
    )
    handoff = traditional_report["authoritative_analysis_snapshot"]
    assert handoff["snapshot_path"] == str(report_path)
    assert handoff["contract_version"] == CONTRACT_VERSION
    assert handoff["optimizer_metadata"]["method"] == "test"
    assert handoff["snapshot_sha256"] == hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    for name, artifact in reev.result.files.items():
        artifact = artifact.resolve()
        if artifact.suffix.lower() != ".json" or artifact == report_path.resolve():
            continue
        artifact_payload = json.loads(artifact.read_text(encoding="utf-8"))
        assert (
            artifact_payload["authoritative_analysis_snapshot"][
                "snapshot_sha256"
            ]
            == handoff["snapshot_sha256"]
        ), name
    json.dumps(reev.authoritative_snapshot.to_dict(), allow_nan=False)


def test_zero_film_structural_and_thermal_profiles_remain_comparable(
    mission, design_vector
):
    zero_film_design = replace(design_vector, film_frac=0.0)
    state = solve_engine_state(zero_film_design, mission)
    design = {
        key: float(value)
        for key, value in zero_film_design.as_dict().items()
    }
    reev = reevaluate(
        design,
        mission,
        mdo_result=state,
        mdo_summary={"Isp": float(state.performance.Isp_delivered)},
        size_pumps=False,
    )

    for field_name in ("all_constraints_feasible", "physics_feasible"):
        field = reev.authoritative_snapshot.constraints_gates[field_name]
        assert not field.available
        assert "electric-pump sizing was not supplied" in (
            field.availability_reason or ""
        )

    # Bound every common scalar/profile in the categories that the old tests
    # merely checked for computability.  These ceilings are above the measured
    # cross-model discrepancy (the models are not algebraic mirrors), but low
    # enough to catch large thermal, structural, cooling, or injector drift.
    category_ceilings = {
        "thermal": 0.45,
        "cooling": 0.20,
        "injector": 1.0e-8,
    }
    bounded_counts = {name: 0 for name in category_ceilings}
    for comparisons in (
        reev.comparison.scalars,
        reev.comparison.profiles,
    ):
        for path, comparison in comparisons.items():
            category = path.split(".", 1)[0]
            if category not in category_ceilings or comparison.not_comparable:
                continue
            if comparison.relative_delta is None:
                # Non-numeric common convention fields must still agree exactly.
                assert comparison.status == "within_tolerance", path
                continue
            assert np.isfinite(comparison.relative_delta), path
            assert comparison.relative_delta <= category_ceilings[category], path
            bounded_counts[category] += 1
    assert bounded_counts["thermal"] >= 10
    assert bounded_counts["cooling"] >= 10
    assert bounded_counts["injector"] >= 10


def test_custom_efficiency_material_and_feed_inputs_survive_real_reevaluation(
    mission, design_vector
):
    custom = replace(
        mission,
        eta_cstar=0.91,
        eta_CF=0.87,
        rho_fuel=799.0,
        rho_ox=1127.0,
        p_vapor_fuel=4200.0,
        p_vapor_ox=88_000.0,
        liner_sigma_allow=123.0e6,
        liner_structural_fos=1.7,
    )
    state = solve_engine_state(design_vector, custom)
    design = {
        key: float(value) for key, value in design_vector.as_dict().items()
    }
    reev = reevaluate(
        design,
        custom,
        mdo_result=state,
        mdo_summary={"Isp": float(state.performance.Isp_delivered)},
        size_pumps=False,
    )
    perf = reev.result.performance
    assert perf.eta_cstar == pytest.approx(float(state.performance.eta_cstar))
    assert perf.eta_CF == pytest.approx(custom.eta_CF)
    assert reev.result.input.material.yield_strength == pytest.approx(
        custom.liner_sigma_allow * custom.liner_structural_fos
    )
    assert reev.result.input.material.structural_fos == pytest.approx(1.7)
    feed = reev.result.report_sections["injector"]["feed"]
    assert feed["fuel"]["density_kg_m3"] == pytest.approx(custom.rho_fuel)
    assert feed["oxidizer"]["density_kg_m3"] == pytest.approx(custom.rho_ox)
    assert feed["fuel"]["vapor_pressure_pa"] == pytest.approx(
        custom.p_vapor_fuel
    )
    assert feed["oxidizer"]["vapor_pressure_pa"] == pytest.approx(
        custom.p_vapor_ox
    )
    for path in (
        "performance.eta_cstar",
        "performance.eta_cf",
        "feed_electrical.fuel_density_kg_m3",
        "feed_electrical.oxidizer_density_kg_m3",
        "feed_electrical.fuel_vapor_pressure_pa",
        "feed_electrical.oxidizer_vapor_pressure_pa",
    ):
        assert reev.comparison.scalars[path].status == "within_tolerance"


def test_hardware_mass_ledger_closes_on_both_paths_and_is_geometry_traceable(
    mission, design_vector, engine_state
):
    """The chamber/interface/injector mass ledger must exist on the
    authoritative side, and its disagreement with the MDO integral must be
    attributable to geometry rather than to two different mass models.
    """

    design = {k: float(v) for k, v in design_vector.as_dict().items()}
    reev = reevaluate(
        design,
        mission,
        mdo_result=engine_state,
        mdo_summary={"Isp": float(engine_state.performance.Isp_delivered)},
        size_pumps=False,
    )

    hardware = reev.result.report_sections["hardware_mass"]
    assert hardware["status"] == "resolved"
    assert hardware["complete"] is True
    assert hardware["total_mass_kg"] > 0.0
    # Every subsystem the "requirements in, parts out" workflow exports must
    # be priced: chamber wall, bolted interface, injector.
    assert set(hardware["by_subsystem_kg"]) == {
        "thrust_chamber", "chamber_interface", "injector",
    }
    assert all(v > 0.0 for v in hardware["by_subsystem_kg"].values())
    # The closeout assumption must be stated, not silently applied.
    assert any("closeout" in note for note in hardware["notes"])
    # Nothing may claim to be qualified hardware.
    assert any(
        item["status"] == "screening_sized" for item in hardware["items"]
    )

    snap = reev.authoritative_snapshot
    branches = (
        "thrust_chamber_liner_mass_kg",
        "thrust_chamber_land_mass_kg",
        "thrust_chamber_closeout_mass_kg",
    )
    for key in branches:
        assert snap.masses[key].available
    assert snap.masses["thrust_chamber_mass_kg"].value == pytest.approx(
        sum(snap.masses[k].value for k in branches), rel=1e-12
    )
    assert snap.masses["injector_mass_kg"].available

    # Both paths must be comparable now -- that is the whole point of closing
    # the gap -- and both must price the SAME alloy, so any residual delta is
    # geometry, not materials.
    comparison = reev.comparison.scalars["masses.thrust_chamber_mass_kg"]
    assert not comparison.not_comparable
    assert np.isfinite(comparison.relative_delta)

    idx = {name: i for i, name in enumerate(MASS_FIELD_NAMES)}
    mdo_total = float(
        engine_state.masses.values[idx["thrust_chamber_mass"]]
    )
    trad_total = snap.masses["thrust_chamber_mass_kg"].value

    # The two shells are integrated over different meridians (the MDO's
    # analytic chamber length vs the traditional L*-derived chamber), so the
    # mass ratio must track the WETTED-AREA ratio.  If it ever stops doing so,
    # the mass models themselves have diverged and this test should fail.
    x_mdo = np.asarray(engine_state.geometry.x, dtype=float)
    r_mdo = np.asarray(engine_state.geometry.r, dtype=float)
    x_trad = np.asarray(reev.result.contour["x"], dtype=float)
    r_trad = np.asarray(reev.result.contour["y"], dtype=float)

    def wetted(x, r):
        seg = np.hypot(np.diff(x), np.diff(r))
        w = np.empty(len(seg) + 1)
        w[0], w[-1] = 0.5 * seg[0], 0.5 * seg[-1]
        w[1:-1] = 0.5 * (seg[:-1] + seg[1:])
        return float(np.sum(2.0 * np.pi * r * w))

    area_ratio = wetted(x_trad, r_trad) / wetted(x_mdo, r_mdo)
    assert trad_total / mdo_total == pytest.approx(area_ratio, rel=0.03)


def test_chamber_geometry_convention_is_shared_by_both_pipelines(
    mission, design_vector, engine_state
):
    """R0: the MDO and traditional chambers must be the SAME chamber.

    Until 2026-07-31 the MDO used a prescribed barrel length ``L*/CR`` while
    the traditional path root-solved the barrel so the injector-face-to-throat
    volume equalled ``L*.A_t``.  NASA SP-125 (printed p. 88) defines the
    chamber volume as spanning injector face to throat plane, so the convergent
    section carries part of it and ``L*/CR`` makes the barrel too long -- by
    20.1 mm and 11.7% of wetted area at this baseline.  An 11.7% wetted-area
    error is an 11.7% error in total heat load, so this test is a physics gate,
    not a bookkeeping one.
    """

    design = {k: float(v) for k, v in design_vector.as_dict().items()}
    reev = reevaluate(
        design,
        mission,
        mdo_result=engine_state,
        mdo_summary={"Isp": float(engine_state.performance.Isp_delivered)},
        size_pumps=False,
    )
    mdo = snapshot_from_mdo(engine_state, mission=mission)
    trad = reev.authoritative_snapshot

    for key in (
        "chamber_barrel_length_m",
        "chamber_volume_m3",
        "chamber_volume_target_m3",
        "wetted_area_m2",
    ):
        assert mdo.geometry[key].available, key
        assert trad.geometry[key].available, key

    # The barrel length and chamber volume are solved from the same closure by
    # the same construction, so they must agree to solver tolerance.
    assert mdo.geometry["chamber_barrel_length_m"].value == pytest.approx(
        trad.geometry["chamber_barrel_length_m"].value, rel=1e-6
    )
    assert mdo.geometry["chamber_volume_m3"].value == pytest.approx(
        trad.geometry["chamber_volume_m3"].value, rel=1e-6
    )

    # Both must actually hit L*.A_t.
    for snap in (mdo, trad):
        assert snap.geometry["chamber_volume_m3"].value == pytest.approx(
            snap.geometry["chamber_volume_target_m3"].value, rel=1e-6
        )

    # Wetted area may differ only by station-grid discretisation: the MDO
    # marches ~24 stations against the traditional contour's several hundred,
    # so the coarse polyline chord-cuts the fillet and throat arcs slightly.
    # 1% catches any return of a convention mismatch while tolerating that.
    area_ratio = (
        trad.geometry["wetted_area_m2"].value
        / mdo.geometry["wetted_area_m2"].value
    )
    assert area_ratio == pytest.approx(1.0, abs=0.01)

    # The barrel must be materially shorter than the old L*/CR approximation,
    # which ignored the convergent section's share of the chamber volume.
    naive = mission.l_star / mission.contraction_ratio
    assert mdo.geometry["chamber_barrel_length_m"].value < 0.5 * naive


def test_chamber_volume_margin_is_a_reported_constraint(mission, engine_state):
    """A chamber whose fixed sections already exceed L*.A_t is infeasible, and
    must be reported as such rather than clamped to a positive barrel."""

    from raosim.mdo.grid import chamber_barrel_length
    from raosim.mdo.state import ENGINE_CONSTRAINT_NAMES

    idx = ENGINE_CONSTRAINT_NAMES.index("chamber_volume_margin")
    margin = float(engine_state.constraints.values[idx])
    assert margin > 0.0
    assert margin == pytest.approx(
        float(engine_state.geometry.chamber_length), rel=1e-12
    )

    # Starve L* until the shoulder/convergent/arc alone overfill the chamber.
    starved = replace(mission, l_star=0.05)
    Rt = float(engine_state.performance.Rt)
    assert float(chamber_barrel_length(Rt, starved)) < 0.0
