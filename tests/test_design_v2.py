from datetime import date
import hashlib
import json
import sys
import types

import pytest

from raosim.design import (
    CoolingSpec,
    DesignInput,
    HostRaoSolverSpec,
    InterfaceSpec,
    ManufacturingSpec,
    MaterialSpec,
    MissionAmbientSpec,
    ThermoSpec,
    design_nozzle_v2,
)
from raosim.nozzle_geometry import bell_nozzle_contour
from raosim.injector import (
    FeedLineSpec,
    FeedSystemSpec,
    InjectorSpec,
    PintleGeometrySpec,
    PropellantFeedSpec,
)
from raosim.physics import bartz_heat_flux, boundary_layer_displacement, regenerative_cooling_screen, structural_screen
from raosim.coolants import canonical_coolant_name
from raosim.physics import (
    resolve_coolant_inlet_temperature,
    resolve_coolant_properties,
)
from raosim.propellants import get_propellant
from raosim.release_readiness import evidence_manifest_template
from raosim.spray_coupling import SprayCStarCouplingSpec
from raosim.throat_geometry import upstream_radius_ratio_for_discharge_coefficient


def _write_frozen_lox_rp1_table(
    tmp_path,
    *,
    pressure_pa=8.0e6,
    temperature_k=3571.0,
    mixture_ratio=2.27,
    molecular_weight=0.0219,
):
    from raosim.frozen_flow import MODEL_ID

    source_payload = b"manufactured frozen LOX/RP-1 design-v2 fixture"
    payload = {
        "schema_version": 2,
        "model": MODEL_ID,
        "molecular_weight_kg_mol": molecular_weight,
        "composition_mass_fractions": {"manufactured_products": 1.0},
        "temperature_nodes_k": [150.0, 500.0, 1000.0, 1800.0, 2600.0, 3300.0, 3800.0],
        "cp_nodes_j_kg_k": [1180.0, 1300.0, 1480.0, 1710.0, 1900.0, 2040.0, 2120.0],
        "source": "manufactured design-v2 variable-cp regression fixture",
        "freeze_basis": "chamber_equilibrium_snapshot",
        "composition_state_pressure_pa": pressure_pa,
        "composition_state_temperature_k": temperature_k,
        "mixture_ratio": mixture_ratio,
        "generator": "pytest manufactured table builder",
        "generator_version": "1",
        "thermo_database": "manufactured piecewise-linear cp",
        "source_artifact_sha256": hashlib.sha256(source_payload).hexdigest(),
    }
    path = tmp_path / "frozen_lox_rp1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _prelim_input(**kwargs):
    base = DesignInput(
        thermo=ThermoSpec(
            mode="constant_gamma",
            propellant_name="LOX/RP-1",
        ),
        Pc=80e5,
        Rt=0.020,
        epsilon=8.0,
        mode="preliminary",
        method="bezier",
    )
    for key, value in kwargs.items():
        setattr(base, key, value)
    return base


def _install_fake_rocketcea(monkeypatch):
    class FakeCEA:
        def __init__(self, **_kwargs):
            pass

        def get_Chamber_MolWt_gamma(self, Pc, MR):
            return 22.0, 1.21

        def get_Tcomb(self, Pc, MR):
            return 3600.0

        def get_Cstar(self, Pc, MR):
            return 1750.0

    rocketcea_mod = types.ModuleType("rocketcea")
    cea_units_mod = types.ModuleType("rocketcea.cea_obj_w_units")
    cea_units_mod.CEA_Obj = FakeCEA
    monkeypatch.setitem(sys.modules, "rocketcea", rocketcea_mod)
    monkeypatch.setitem(sys.modules, "rocketcea.cea_obj_w_units", cea_units_mod)


def test_preliminary_v2_constant_gamma_runs():
    result = design_nozzle_v2(_prelim_input())

    assert result.design_status == "preliminary_top_geometry"
    assert result.report_sections["thermochemistry"]["source"] == "built_in_constant_gamma"
    assert result.report_sections["boundary_layer"]["effective_epsilon"] < result.input.epsilon
    assert result.contour["hardware_qualified"] is False
    assert result.hardware_qualified is False
    assert result.report_sections["physical_release_readiness"]["blocked"] is True
    assert result.report_sections["model_registry_audit"]["passed"] is True


def test_v2_explicit_efficiency_split_survives_constant_gamma():
    database_propellant = get_propellant("LOX/RP-1")
    database_efficiencies = (
        float(database_propellant.eta_cstar),
        float(database_propellant.eta_CF),
    )
    request = _prelim_input(
        thermo=ThermoSpec(
            mode="constant_gamma",
            propellant_name="LOX/RP-1",
            eta_Isp=0.50,  # Explicit split below is authoritative.
            eta_cstar=0.91,
            eta_CF=0.87,
        )
    )

    result = design_nozzle_v2(request)
    performance = result.performance

    assert result.propellant.eta_cstar == pytest.approx(0.91)
    assert result.propellant.eta_CF == pytest.approx(0.87)
    assert result.propellant.eta_Isp == pytest.approx(0.91 * 0.87)
    assert performance.eta_cstar == pytest.approx(0.91)
    assert performance.eta_CF == pytest.approx(0.87)
    assert performance.eta_Isp == pytest.approx(0.91 * 0.87)
    assert performance.c_star_effective == pytest.approx(
        performance.c_star * 0.91
    )
    assert performance.Cf_actual == pytest.approx(
        performance.Cf_ideal * 0.87
    )
    assert result.thermochemistry.chamber_state["eta_cstar"] == pytest.approx(
        0.91
    )
    assert result.thermochemistry.chamber_state["eta_CF"] == pytest.approx(
        0.87
    )
    # Resolving one design must not mutate the shared database entry.
    assert database_propellant.eta_cstar == pytest.approx(
        database_efficiencies[0]
    )
    assert database_propellant.eta_CF == pytest.approx(
        database_efficiencies[1]
    )


@pytest.mark.parametrize(
    ("efficiency_kwargs", "expected_eta_cstar", "expected_eta_cf"),
    (
        ({"eta_Isp": 0.83}, 1.0, 0.83),
        (
            {"eta_Isp": 0.50, "eta_cstar": 0.92, "eta_CF": 0.88},
            0.92,
            0.88,
        ),
    ),
)
def test_v2_cea_preserves_legacy_or_explicit_efficiency_convention(
    monkeypatch,
    efficiency_kwargs,
    expected_eta_cstar,
    expected_eta_cf,
):
    _install_fake_rocketcea(monkeypatch)
    request = _prelim_input(
        thermo=ThermoSpec(
            mode="cea_frozen",
            propellant_name="LOX/RP-1",
            oxidizer="LOX",
            fuel="RP-1",
            mixture_ratio=2.6,
            **efficiency_kwargs,
        )
    )

    result = design_nozzle_v2(request)
    performance = result.performance

    assert result.propellant.c_star == pytest.approx(1750.0)
    assert result.propellant.eta_cstar == pytest.approx(expected_eta_cstar)
    assert result.propellant.eta_CF == pytest.approx(expected_eta_cf)
    assert performance.eta_cstar == pytest.approx(expected_eta_cstar)
    assert performance.eta_CF == pytest.approx(expected_eta_cf)
    assert performance.eta_Isp == pytest.approx(
        expected_eta_cstar * expected_eta_cf
    )
    assert performance.c_star_effective == pytest.approx(
        1750.0 * expected_eta_cstar
    )
    assert performance.Cf_actual == pytest.approx(
        performance.Cf_ideal * expected_eta_cf
    )


def test_v2_frozen_variable_cp_bezier_uses_profile_and_fails_validation_gate(
    tmp_path,
):
    table = _write_frozen_lox_rp1_table(tmp_path)
    request = _prelim_input(
        thermo=ThermoSpec(
            mode="constant_gamma",
            propellant_name="LOX/RP-1",
            mixture_ratio=2.27,
            expansion_model="frozen_variable_cp",
            frozen_gas_table=table,
        )
    )

    result = design_nozzle_v2(request)
    frozen = result.performance.frozen_flow

    assert frozen is not None
    assert frozen.all_closures_pass is True
    assert result.performance.expansion_model == "frozen_variable_cp_q1d"
    assert result.performance.Me == pytest.approx(frozen.exit.mach)
    assert result.contour["ideal_exit_Mach"] == pytest.approx(frozen.exit.mach)
    assert result.contour["quasi_1d_expansion_model"] == (
        "frozen_variable_cp_q1d"
    )
    thermo_report = result.report_sections["thermochemistry"]
    assert thermo_report["frozen_expansion"]["closures"]["all_pass"] is True
    assert thermo_report["exit_state"]["property_table"]["freeze_basis"] == (
        "chamber_equilibrium_snapshot"
    )
    gates = {check.name: check for check in result.gate_report.checks}
    assert gates["frozen_variable_cp_q1d_closure"].passed is True
    assert gates["frozen_property_and_performance_benchmark"].passed is False
    assert gates["variable_property_boundary_layer"].passed is False
    assert gates["variable_property_bartz_recovery"].passed is False
    assert gates["variable_property_discharge_coefficient"].passed is False
    serialized = result.to_dict()["performance"]
    assert serialized["expansion_model"] == "frozen_variable_cp_q1d"
    assert serialized["frozen_flow_fingerprint"] == (
        result.performance.frozen_flow_fingerprint
    )


def test_v2_frozen_variable_cp_closes_target_thrust_and_matched_pressure(
    tmp_path, monkeypatch
):
    import raosim.design as design_module

    table = _write_frozen_lox_rp1_table(tmp_path)

    def constant_gamma_inverse_forbidden(*_args, **_kwargs):
        raise AssertionError("constant-gamma pressure inverse was called")

    monkeypatch.setattr(
        design_module,
        "expansion_ratio_from_pressure",
        constant_gamma_inverse_forbidden,
    )
    request = _prelim_input(
        Rt=None,
        target_thrust=13_000.0,
        epsilon=None,
        thermo=ThermoSpec(
            mode="constant_gamma",
            propellant_name="LOX/RP-1",
            mixture_ratio=2.27,
            expansion_model="frozen_variable_cp",
            frozen_gas_table=table,
        ),
    )

    result = design_nozzle_v2(request)

    assert result.performance.thrust == pytest.approx(13_000.0, rel=2e-12)
    assert result.performance.Pe == pytest.approx(
        request.ambient.design_pressure, rel=5e-10
    )
    assert result.input.Rt is not None
    assert result.input.epsilon > 1.0


@pytest.mark.parametrize("method", ("moc", "rao", "rao_variational_moc"))
def test_v2_frozen_variable_cp_rejects_constant_gamma_characteristic_methods(
    tmp_path, method
):
    table = _write_frozen_lox_rp1_table(tmp_path)
    request = _prelim_input(
        method=method,
        thermo=ThermoSpec(
            propellant_name="LOX/RP-1",
            expansion_model="frozen_variable_cp",
            frozen_gas_table=table,
        ),
    )
    with pytest.raises(ValueError, match="compatible only with bezier"):
        design_nozzle_v2(request)


def test_v2_characteristic_contour_receives_design_ambient_ratio(monkeypatch):
    import raosim.design as design_module

    captured = {}
    original = design_module.bell_nozzle_contour

    def capture_then_build_bezier(*args, **kwargs):
        captured["pa_over_p0"] = kwargs["pa_over_p0"]
        kwargs["method"] = "bezier"
        return original(*args, **kwargs)

    monkeypatch.setattr(
        design_module, "bell_nozzle_contour", capture_then_build_bezier
    )
    request = _prelim_input(
        Pc=2.0e6,
        method="rao_variational_moc",
        ambient=MissionAmbientSpec(Pa=80_000.0),
    )
    design_module._build_v2_contour(
        request,
        Rt=0.020,
        epsilon=8.0,
        prop=get_propellant("LOX/RP-1"),
    )

    assert captured["pa_over_p0"] == pytest.approx(0.04)


def test_v2_characteristic_contour_receives_host_solver_controls(monkeypatch):
    import raosim.design as design_module

    captured = {}
    original = design_module.bell_nozzle_contour

    def capture_then_build_bezier(*args, **kwargs):
        captured.update(kwargs)
        kwargs["method"] = "bezier"
        return original(*args, **kwargs)

    monkeypatch.setattr(
        design_module, "bell_nozzle_contour", capture_then_build_bezier
    )
    request = _prelim_input(
        method="rao_variational_moc",
        host_rao_solver=HostRaoSolverSpec(
            n_control=18,
            n_kernel=20,
            max_nfev=321,
            evaluate_moc=False,
            theta_n_guess_deg=33.0,
            starting_line_method="area_ratio",
            solver_backend="numpy",
            wall_method="bde",
            kernel_d_fraction_max=0.7,
            physics_weight=1.0,
        ),
    )

    design_module._build_v2_contour(
        request,
        Rt=0.020,
        epsilon=8.0,
        prop=get_propellant("LOX/RP-1"),
    )

    assert captured["rao_moc_n_control"] == 18
    assert captured["rao_moc_n_kernel"] == 20
    assert captured["rao_moc_max_nfev"] == 321
    assert captured["rao_moc_evaluate_moc"] is False
    assert captured["rao_moc_theta_n_guess_deg"] == pytest.approx(33.0)
    assert captured["starting_line_method"] == "area_ratio"
    assert captured["rao_moc_solver_backend"] == "numpy"
    assert captured["rao_moc_wall_method"] == "bde"
    assert captured["rao_moc_kernel_d_fraction_max"] == pytest.approx(0.7)
    assert captured["rao_moc_physics_weight"] == pytest.approx(1.0)


def test_v2_uses_scheduled_design_ambient_consistently():
    request = _prelim_input(
        ambient=MissionAmbientSpec(
            Pa=101_325.0,
            altitude_schedule_m=[0.0, 10_000.0],
            pressure_schedule_pa=[101_325.0, 26_500.0],
        )
    )

    result = design_nozzle_v2(request)

    assert request.ambient.design_pressure == pytest.approx(26_500.0)
    assert result.performance.Pa == pytest.approx(26_500.0)
    from raosim.mdo.snapshot import snapshot_from_traditional

    snapshot = snapshot_from_traditional(result)
    assert snapshot.performance["ambient_pressure_pa"].value == pytest.approx(
        26_500.0
    )
    assert snapshot.provenance["input_conventions"].value[
        "ambient_pressure_pa"
    ] == pytest.approx(26_500.0)


@pytest.mark.parametrize(
    "ambient",
    (
        MissionAmbientSpec(Pa=-1.0),
        MissionAmbientSpec(Pa=8.0e6),
        MissionAmbientSpec(Pa=101_325.0, pressure_schedule_pa=[-1.0]),
    ),
)
def test_v2_rejects_invalid_ambient_pressures(ambient):
    with pytest.raises(ValueError, match="ambient pressures"):
        design_nozzle_v2(_prelim_input(ambient=ambient))


def test_v2_frozen_variable_cp_requires_explicit_consistent_property_evidence(
    tmp_path,
):
    missing = _prelim_input(
        thermo=ThermoSpec(
            propellant_name="LOX/RP-1",
            expansion_model="frozen_variable_cp",
        )
    )
    with pytest.raises(ValueError, match="requires thermo.frozen_gas_table"):
        design_nozzle_v2(missing)

    table = _write_frozen_lox_rp1_table(tmp_path)
    stale_constant = _prelim_input(
        thermo=ThermoSpec(
            propellant_name="LOX/RP-1",
            frozen_gas_table=table,
        )
    )
    with pytest.raises(ValueError, match="requires expansion_model"):
        design_nozzle_v2(stale_constant)

    wrong_of = _prelim_input(
        thermo=ThermoSpec(
            propellant_name="LOX/RP-1",
            mixture_ratio=2.5,
            expansion_model="frozen_variable_cp",
            frozen_gas_table=table,
        )
    )
    with pytest.raises(ValueError, match="mixture_ratio does not match"):
        design_nozzle_v2(wrong_of)


def test_v2_frozen_variable_cp_rejects_stale_state_and_validated_claim(
    tmp_path,
):
    stale_table = _write_frozen_lox_rp1_table(
        tmp_path, pressure_pa=7.5e6
    )
    stale = _prelim_input(
        thermo=ThermoSpec(
            propellant_name="LOX/RP-1",
            mixture_ratio=2.27,
            expansion_model="frozen_variable_cp",
            frozen_gas_table=stale_table,
        )
    )
    with pytest.raises(ValueError, match="snapshot pressure"):
        design_nozzle_v2(stale)

    table = _write_frozen_lox_rp1_table(tmp_path)
    validated = _prelim_input(
        mode="validated",
        thermo=ThermoSpec(
            propellant_name="LOX/RP-1",
            mixture_ratio=2.27,
            expansion_model="frozen_variable_cp",
            frozen_gas_table=table,
        ),
    )
    with pytest.raises(ValueError, match="validated mode does not yet accept"):
        design_nozzle_v2(validated)


def test_v2_release_evidence_hard_gate_requires_a_manifest():
    with pytest.raises(ValueError, match="release_evidence_manifest"):
        design_nozzle_v2(_prelim_input(require_release_evidence=True))
    with pytest.raises(ValueError, match="configuration_id"):
        design_nozzle_v2(_prelim_input(
            require_release_evidence=True,
            release_evidence_manifest="evidence.json",
        ))


def test_v2_release_gate_matches_configuration_but_never_qualifies_hardware(
    tmp_path,
):
    configuration_id = "ENGINE-CFG-UNIT-TEST"
    manifest = evidence_manifest_template("engine")
    manifest["configuration_id"] = configuration_id
    for record in manifest["evidence"]:
        record.update({
            "passed": True,
            "artifact": "archive://qualification/" + record["requirement_id"],
            "artifact_sha256": "a" * 64,
            "configuration_id": configuration_id,
            "reviewed_by": "independent test authority",
            "review_date": date.today().isoformat(),
        })
    path = tmp_path / "engine_evidence.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    result = design_nozzle_v2(_prelim_input(
        configuration_id=configuration_id,
        release_evidence_manifest=path,
        require_release_evidence=True,
    ))
    release = result.report_sections["physical_release_readiness"]
    assert release["evidence_complete"] is True
    assert release["blocked"] is False
    assert release["hardware_qualified"] is False
    assert result.hardware_qualified is False


def test_equilibrium_cea_mode_is_rejected_until_expansion_is_supported():
    from raosim.cea import resolve_thermochemistry

    with pytest.raises(NotImplementedError, match="cea_equilibrium"):
        resolve_thermochemistry(
            thermo_mode="cea_equilibrium",
            propellant_name="LOX/RP-1",
            Pc=80e5,
            mixture_ratio=2.6,
            oxidizer="LOX",
            fuel="RP-1",
            epsilon=8.0,
        )


def test_v2_derives_throat_radius_from_cd_target_and_auto_shoulder():
    result = design_nozzle_v2(_prelim_input(throat_cd_target=0.989))

    expected_ru = upstream_radius_ratio_for_discharge_coefficient(
        0.989, result.propellant.gamma
    )
    chamber = result.report_sections["chamber_geometry"]

    assert result.input.throat_geometry.upstream_radius_ratio == pytest.approx(
        expected_ru
    )
    assert result.input.shoulder_radius_factor is not None
    assert result.input.shoulder_radius_factor != pytest.approx(0.25)
    assert chamber["shoulder_radius_source"] == "auto_geometric_closure"
    assert chamber["throat_upstream_radius_source"] == "cd_target_hall_sp8120"
    assert chamber["throat_discharge_coefficient_hall"] == pytest.approx(0.989)


def test_throat_radius_beyond_sp8120_requires_explicit_extension():
    with pytest.raises(ValueError, match="SP-8120 range capability"):
        design_nozzle_v2(_prelim_input(throat_cd_target=0.99))

    result = design_nozzle_v2(_prelim_input(
        throat_cd_target=0.99,
        allow_throat_radius_extension=True,
    ))
    chamber = result.report_sections["chamber_geometry"]
    assert chamber["throat_upstream_radius_source"] == (
        "cd_target_hall_repository_extension"
    )
    assert result.input.throat_geometry.upstream_radius_ratio > 1.5


def test_v2_backend_evaluates_requested_pintle():
    result = design_nozzle_v2(_prelim_input(
        injector=InjectorSpec(type="pintle"),
    ))

    injector = result.report_sections["injector"]
    assert injector["feasible"] is True
    assert injector["feed"]["fuel"]["name"] == "RP-1"
    assert injector["feed"]["oxidizer"]["name"] == "LOX"
    assert any(
        check.category == "injector"
        for check in result.gate_report.checks
    )
    expected_length = (
        result.contour["throat_location"]
        - result.contour["injector_location"]
    )
    assert result.report_sections["injector"]["chamber_length_m"] == pytest.approx(
        expected_length
    )
    assert result.report_sections["chamber_geometry"][
        "injector_to_throat_length"
    ] == pytest.approx(expected_length)


def test_v2_explicit_pintle_cad_failure_is_a_hard_failure(
    tmp_path, monkeypatch
):
    import raosim.injector_export as injector_export

    def fail_export(*_args, **_kwargs):
        raise RuntimeError("synthetic CAD topology failure")

    monkeypatch.setattr(
        injector_export, "export_pintle_package", fail_export
    )
    request = _prelim_input(
        injector=InjectorSpec(type="pintle", cad="machined"),
        manufacturing=ManufacturingSpec(
            output_dir=tmp_path,
            cad="none",
        ),
    )
    with pytest.raises(RuntimeError, match="Requested pintle CAD failed"):
        design_nozzle_v2(request)
    error = tmp_path / "pintle" / "EXPORT_ERROR.txt"
    assert error.exists()
    assert "synthetic CAD topology failure" in error.read_text()


def test_v2_diagnostic_only_pintle_package_can_remain_best_effort(
    tmp_path, monkeypatch
):
    import raosim.injector_export as injector_export

    def fail_export(*_args, **_kwargs):
        raise RuntimeError("synthetic plot failure")

    monkeypatch.setattr(
        injector_export, "export_pintle_package", fail_export
    )
    result = design_nozzle_v2(_prelim_input(
        injector=InjectorSpec(type="pintle", cad="none"),
        manufacturing=ManufacturingSpec(
            output_dir=tmp_path,
            cad="none",
        ),
    ))
    assert result.files["pintle_error"].exists()


def test_v2_opt_in_spray_cstar_loop_closes_injector_and_cycle_mass_flow():
    result = design_nozzle_v2(_prelim_input(
        Pc=1.5e6,
        injector=InjectorSpec(
            type="pintle",
            fuel_dp_fraction=0.7,
            oxidizer_dp_fraction=0.7,
            evaporation_constant=1.0e-4,
        ),
        spray_cstar_coupling=SprayCStarCouplingSpec(
            enabled=True,
            eta_mixing=0.98,
            eta_combustion=0.99,
            relative_tolerance=1.0e-5,
        ),
    ))

    coupling = result.report_sections["spray_cstar_coupling"]
    injector = result.report_sections["injector"]
    injector_mdot = (
        injector["annulus"]["mdot_kg_s"] + injector["slots"]["mdot_kg_s"]
    )

    assert coupling["converged"] is True
    assert coupling["iteration_count"] > 0
    assert result.performance.eta_cstar == pytest.approx(coupling["eta_cstar"])
    assert result.performance.m_dot == pytest.approx(
        coupling["required_mass_flow_kg_s"]
    )
    assert injector_mdot == pytest.approx(result.performance.m_dot)
    assert result.report_sections["thrust_closure"][
        "spray_cstar_fixed_point_enabled"
    ] is True


def test_v2_spray_cstar_regen_outer_loop_recloses_cooling_and_pump_duty():
    request = _prelim_input(
        Pc=1.5e6,
        thermo=ThermoSpec(
            mode="constant_gamma",
            propellant_name="N2O/Ethanol",
            mixture_ratio=1.0,
        ),
        injector=InjectorSpec(
            type="pintle",
            allow_infeasible=True,
            fuel_dp_fraction=0.9,
            oxidizer_dp_fraction=0.7,
            evaporation_constant=1.0e-4,
            feed_system=FeedSystemSpec(
                fuel=FeedLineSpec(tank_pressure=1.0e5),
                oxidizer=FeedLineSpec(tank_pressure=1.0e5),
            ),
            oxidizer=PropellantFeedSpec(
                role="oxidizer",
                name="N2O",
                inlet_temperature=270.0,
                inlet_pressure=4.0e6,
                phase="liquid",
                density=900.0,
                viscosity=1.0e-4,
                surface_tension=0.02,
                vapor_pressure=2.0e6,
                property_source="explicit test-state properties",
            ),
        ),
        cooling=CoolingSpec(
            method="regenerative",
            coolant="ethanol",
            coolant_inlet_temperature=220.0,
            coolant_mass_flow=None,
            channel_count=40,
            channel_width=8.0e-4,
            channel_height=2.5e-3,
            coolant_property_backend="constant",
        ),
        manufacturing=ManufacturingSpec(wall_thickness=1.0e-3),
        spray_cstar_coupling=SprayCStarCouplingSpec(
            enabled=True,
            eta_mixing=0.98,
            eta_combustion=0.99,
            relative_tolerance=1.0e-5,
        ),
    )
    result = design_nozzle_v2(request)
    coupling = result.report_sections["spray_cstar_coupling"]
    final = coupling["final_state_summary"]
    assert coupling["converged"] is True
    assert final["total_mass_flow_kg_s"] == pytest.approx(
        result.performance.m_dot
    )
    assert final["coolant_mass_flow_kg_s"] == pytest.approx(
        final["fuel_mass_flow_kg_s"]
    )
    assert final["regen_fuel_relative_flow_error"] == pytest.approx(0.0)
    assert final["coolant_pressure_drop_pa"] == pytest.approx(
        result.report_sections["cooling"]["coolant_pressure_drop"]
    )
    assert final["feed_and_pump_duty_by_role"]["fuel"][
        "regen_loss_pa"
    ] == pytest.approx(final["coolant_pressure_drop_pa"])
    assert final["feed_and_pump_duty_by_role"]["fuel"][
        "ideal_pump_power_w"
    ] > 0.0
    for iteration in coupling["iterations"]:
        state = iteration["state_summary"]
        assert state["coolant_mass_flow_kg_s"] == pytest.approx(
            state["fuel_mass_flow_kg_s"]
        )
        assert state["feed_and_pump_duty_by_role"]["fuel"][
            "ideal_pump_power_w"
        ] > 0.0


def test_v2_spray_regen_rejects_unmodelled_independent_coolant_split():
    request = _prelim_input(
        injector=InjectorSpec(type="pintle"),
        cooling=CoolingSpec(
            method="regenerative",
            coolant="water",
            channel_count=12,
            channel_width=1.0e-3,
            channel_height=1.0e-3,
        ),
        spray_cstar_coupling=SprayCStarCouplingSpec(
            enabled=True,
            eta_mixing=0.98,
            eta_combustion=0.99,
        ),
    )
    with pytest.raises(ValueError, match="independent coolant/bypass"):
        design_nozzle_v2(request)


def test_v2_rejects_fuel_film_branch_without_fuel_regen_topology():
    request = _prelim_input(
        cooling=CoolingSpec(
            method="regenerative",
            coolant="water",
            channel_count=12,
            channel_width=1.0e-3,
            channel_height=1.0e-3,
            coolant_mass_flow=1.0,
            fuel_film_mass_flow=0.1,
        )
    )

    with pytest.raises(ValueError, match="cycle fuel"):
        design_nozzle_v2(request)


def test_v2_rejects_negative_fuel_film_flow():
    request = _prelim_input(
        cooling=CoolingSpec(fuel_film_mass_flow=-0.1)
    )

    with pytest.raises(ValueError, match="nonnegative"):
        design_nozzle_v2(request)


def test_v2_explicit_film_split_reaches_feed_ledger_without_false_failure():
    baseline = design_nozzle_v2(_prelim_input(Rt=0.10))
    mixture_ratio = float(baseline.propellant.OF)
    total_fuel = baseline.performance.m_dot / (1.0 + mixture_ratio)
    film_flow = 0.15 * total_fuel
    jacket_flow = total_fuel - film_flow
    request = _prelim_input(
        Rt=0.10,
        cooling=CoolingSpec(
            method="regenerative",
            coolant="RP-1",
            channel_count=80,
            channel_width=6.0e-4,
            channel_height=2.5e-3,
            coolant_mass_flow=jacket_flow,
            fuel_film_mass_flow=film_flow,
            coolant_property_backend="constant",
        ),
        injector=InjectorSpec(type="pintle", allow_infeasible=True),
        manufacturing=ManufacturingSpec(wall_thickness=1.0e-3),
    )

    result = design_nozzle_v2(request)
    injector = result.report_sections["injector"]
    closure = next(
        gate for gate in injector["gates"]
        if gate["name"] == "regen_fuel_flow_closure"
    )
    split = injector["feed_system"]["fuel_flow_split"]
    fuel_line = injector["feed_system"]["lines"]["fuel"]

    assert closure["status"] == "pass"
    assert split["total_fuel_mass_flow_kg_s"] == pytest.approx(total_fuel)
    assert split["regen_jacket_mass_flow_kg_s"] == pytest.approx(jacket_flow)
    assert split["film_bypass_mass_flow_kg_s"] == pytest.approx(film_flow)
    assert split["closure_residual_kg_s"] == pytest.approx(0.0, abs=1e-12)
    assert fuel_line["volumetric_flow_m3_s"] == pytest.approx(
        total_fuel / injector["feed"]["fuel"]["density_kg_m3"]
    )


def test_v2_blocks_unbenchmarked_bezier_chart_extrapolation():
    result = design_nozzle_v2(_prelim_input(epsilon=60.0))
    benchmark = result.report_sections["benchmark_status"]

    assert benchmark["validated_for_design"] is False
    assert benchmark["status"] == "unvalidated_rao_chart_extrapolation"
    assert any(
        check.name == "benchmark_status" and not check.passed
        for check in result.gate_report.checks
    )


def test_v2_pintle_resolves_matching_chamber_and_injector_interface():
    request = _prelim_input(
        injector=InjectorSpec(type="pintle"),
        manufacturing=ManufacturingSpec(wall_thickness=0.002),
    )

    result = design_nozzle_v2(request)

    interface = result.input.interface
    geometry = result.input.injector.geometry
    mechanical = result.input.injector.mechanical
    resolution = result.report_sections["injector_interface_resolution"]

    assert interface.flange_od == pytest.approx(interface.injector_face_od)
    assert geometry.face_od == pytest.approx(interface.injector_face_od)
    assert mechanical.faceplate_outer_diameter == pytest.approx(
        interface.injector_face_od
    )
    assert mechanical.bolt_circle_diameter == pytest.approx(
        interface.bolt_circle_diameter
    )
    assert mechanical.bolt_hole_diameter == pytest.approx(
        interface.bolt_hole_diameter
    )
    assert interface.flange_length is not None
    assert resolution["auto_sized_fields"]
    assert result.report_sections["cad_readiness"]["flange_ok"] is True


def test_v2_backend_blocks_infeasible_pintle():
    request = _prelim_input(
        injector=InjectorSpec(
            type="pintle",
            geometry=PintleGeometrySpec(slot_count=400),
        ),
    )
    with pytest.raises(RuntimeError, match="Pintle injector gates failed"):
        design_nozzle_v2(request)


def test_v2_backend_can_report_explicitly_overridden_infeasible_pintle():
    request = _prelim_input(
        injector=InjectorSpec(
            type="pintle",
            allow_infeasible=True,
            geometry=PintleGeometrySpec(slot_count=400),
        ),
    )
    result = design_nozzle_v2(request)
    assert result.report_sections["injector"]["feasible"] is False
    assert result.gate_report.passed is False


def test_cooling_temperature_is_optional_and_resolved_centrally():
    methane = CoolingSpec(coolant="methane")
    hydrogen = CoolingSpec(coolant="lh2")
    explicit = CoolingSpec(coolant="methane", coolant_inlet_temperature=135.0)

    assert methane.coolant_inlet_temperature is None
    assert resolve_coolant_inlet_temperature(methane) == 120.0
    assert resolve_coolant_inlet_temperature(hydrogen) == 25.0
    assert resolve_coolant_inlet_temperature(explicit) == 135.0


@pytest.mark.parametrize("alias", ["methane", "ch4", "lch4"])
def test_methane_aliases_share_temperature_and_properties(alias):
    cooling = CoolingSpec(coolant=alias)
    properties = resolve_coolant_properties(cooling)
    assert canonical_coolant_name(alias) == "methane"
    assert resolve_coolant_inlet_temperature(cooling) == 120.0
    assert properties["rho"] == pytest.approx(423.0)
    assert properties["cp"] == pytest.approx(3450.0)


@pytest.mark.parametrize("alias", ["lh2", "hydrogen", "h2"])
def test_hydrogen_aliases_share_temperature_and_properties(alias):
    cooling = CoolingSpec(coolant=alias)
    properties = resolve_coolant_properties(cooling)
    assert canonical_coolant_name(alias) == "hydrogen"
    assert resolve_coolant_inlet_temperature(cooling) == 25.0
    assert properties["rho"] == pytest.approx(71.0)
    assert properties["cp"] == pytest.approx(9800.0)


def test_v2_backend_uses_central_methane_temperature_default():
    result = design_nozzle_v2(_prelim_input(
        cooling=CoolingSpec(
            method="regenerative",
            coolant="methane",
            channel_count=40,
            channel_width=0.0008,
            channel_height=0.0025,
            coolant_mass_flow=10.0,
            coolant_property_backend="constant",
        ),
        manufacturing=ManufacturingSpec(wall_thickness=0.001),
    ))

    cooling = result.report_sections["cooling"]
    assert cooling["coolant_inlet_temperature"] == 120.0
    assert cooling["coolant_inlet_temperature_source"] == (
        "central_coolant_default"
    )


def test_v2_regen_boundary_uses_fuel_split_not_legacy_scalar():
    result = design_nozzle_v2(_prelim_input(
        cooling=CoolingSpec(
            method="regenerative",
            coolant="rp1",
            channel_count=40,
            channel_width=0.0008,
            channel_height=0.0025,
            coolant_mass_flow=10.0,
            coolant_property_backend="constant",
            injector_pressure_drop=99.0e6,
        ),
        injector=InjectorSpec(type="none", fuel_dp_fraction=0.31),
        manufacturing=ManufacturingSpec(wall_thickness=0.001),
    ))

    cooling = result.report_sections["cooling"]
    assert cooling["coolant_outlet_pressure"] == pytest.approx(
        result.input.Pc * (1.0 + 0.31)
    )
    assert cooling["coolant_pressure_boundary_source"] == (
        "minimum_injector_entry_pressure_Pc_plus_injector_drop"
    )
    assert any(
        "CoolingSpec.injector_pressure_drop is deprecated and ignored" in w
        for w in result.warnings
    )


def test_validated_requires_cea():
    request = _prelim_input(
        mode="validated",
        L_star=1.0,
        contraction_ratio=2.5,
        minimum_cylindrical_length=0.01,
    )

    with pytest.raises(RuntimeError, match="requires CEA"):
        design_nozzle_v2(request)


def test_validated_mode_rejects_placeholder_chamber_inputs():
    request = _prelim_input(
        mode="validated",
        thermo=ThermoSpec(
            mode="cea_frozen",
            propellant_name="LOX/RP-1",
            oxidizer="LOX",
            fuel="RP-1",
            mixture_ratio=2.6,
        ),
    )

    with pytest.raises(ValueError, match="explicit chamber inputs") as exc:
        design_nozzle_v2(request)
    assert "L_star" in str(exc.value)
    assert "contraction_ratio" in str(exc.value)
    assert "minimum_cylindrical_length" in str(exc.value)


def test_experimental_methods_block_validated_and_manufacturing():
    validated = _prelim_input(mode="validated", method="moc")
    with pytest.raises(ValueError, match="bezier"):
        design_nozzle_v2(validated)

    manufacturing = _prelim_input(
        method="rao",
        manufacturing=ManufacturingSpec(wall_thickness=0.002, cad="step"),
    )
    with pytest.raises(ValueError, match="Manufacturing CAD"):
        design_nozzle_v2(manufacturing)


def test_target_thrust_loses_to_explicit_rt():
    request = _prelim_input(target_thrust=5000.0)
    result = design_nozzle_v2(request)

    assert result.input.Rt == pytest.approx(0.020)
    assert any("explicit Rt is used" in warning for warning in result.warnings)
    closure = result.report_sections["thrust_closure"]
    assert closure["rt_sized_from_target_thrust"] is False
    assert closure["target_thrust_N"] == pytest.approx(5000.0)
    assert closure["throat_discharge_coefficient_applied_to_mass_flow"] is False


def test_target_thrust_sizing_reports_quasi_1d_closure_basis():
    request = _prelim_input(Rt=None, target_thrust=5000.0)
    result = design_nozzle_v2(request)

    closure = result.report_sections["thrust_closure"]
    assert closure["rt_sized_from_target_thrust"] is True
    assert closure["calculated_thrust_N"] == pytest.approx(5000.0)
    assert closure["relative_target_error"] == pytest.approx(0.0, abs=1e-12)
    assert closure["contour_audit_Cf"] is None
    assert closure["sizing_basis"].startswith("quasi_1d")


def test_schema_rejects_bad_regen_dimensions():
    request = _prelim_input(
        cooling=CoolingSpec(method="regenerative", channel_count=12)
    )

    with pytest.raises(ValueError, match="channel_width"):
        design_nozzle_v2(request)


def test_v2_step_writes_metadata_and_ipt_is_deferred(tmp_path):
    request = _prelim_input(
        manufacturing=ManufacturingSpec(
            wall_thickness=0.002,
            cad="step",
            output_dir=tmp_path,
        ),
        interface=InterfaceSpec(flange_od=0.20, flange_length=0.010),
    )

    result = design_nozzle_v2(request)

    assert result.files["step"].exists()
    assert result.files["design_report"].exists()
    report = result.files["design_report"].read_text(encoding="utf-8")
    assert '"authoritative_cad": "STEP"' in report
    assert '"native_ipt": "deferred"' in report
    assert '"cad_body_scope": "single_revolved_uniform_wall_body"' in report
    assert result.files["step"].name == "thrust_chamber_wall.step"

    ipt_request = _prelim_input(
        manufacturing=ManufacturingSpec(wall_thickness=0.002, cad="ipt")
    )
    with pytest.raises(ValueError, match="IPT"):
        design_nozzle_v2(ipt_request)


def test_v2_cad_none_reports_no_authoritative_cad_artifact(tmp_path):
    request = _prelim_input(
        manufacturing=ManufacturingSpec(
            wall_thickness=0.002,
            cad="none",
            output_dir=tmp_path,
        )
    )

    result = design_nozzle_v2(request)
    report = json.loads(
        result.files["design_report"].read_text(encoding="utf-8")
    )

    assert "step" not in result.files
    assert report["metadata"]["authoritative_cad"] is None


def test_mocked_rocketcea_feeds_validated_thermochemistry(monkeypatch):
    _install_fake_rocketcea(monkeypatch)

    # A physically survivable regen-cooled copper engine.  Both the
    # throat heat flux (full Bartz, ~58 MW/m² at 70 bar) and the cooling
    # (coupled Sieder-Tate / 1-D conduction) are now real, so the
    # fixture must be a real high-performance regen wall: thin copper,
    # small fast channels (high coolant velocity -> high h_c).  Probed
    # margins: peak gas-side wall ~1020 K, cooling margin ~1.1.  The
    # pre-2026-06 fixture (20 MPa through a 4 mm k=20 wall) only passed
    # because the old screening under-predicted flux ~1000× and the old
    # cooling film model was ad-hoc.  This test checks CEA
    # thermochemistry feed-through, not the thermal margins themselves.
    request = _prelim_input(
        mode="validated",
        thermo=ThermoSpec(
            mode="cea_frozen",
            propellant_name="LOX/RP-1",
            oxidizer="LOX",
            fuel="RP-1",
            mixture_ratio=2.6,
        ),
        Pc=70e5,
        epsilon=4.0,
        L_star=1.0,
        contraction_ratio=2.5,
        minimum_cylindrical_length=0.01,
        manufacturing=ManufacturingSpec(wall_thickness=0.001, cad="none"),
        cooling=CoolingSpec(
            method="regenerative",
            coolant="rp1",
            channel_count=100,           # fits the 20 mm throat (60 of 126 mm)
            channel_width=0.0006,
            channel_height=0.0030,
            coolant_mass_flow=12.0,
            coolant_inlet_temperature=300.0,
            max_wall_temperature=1200.0,
        ),
        material=MaterialSpec(
            yield_strength=2.0e9,
            max_temperature=1300.0,
            max_heat_flux=120e6,
            conductivity=350.0,
        ),
        ambient=MissionAmbientSpec(Pa=5_000.0),
    )

    result = design_nozzle_v2(request)

    assert result.validated is True
    assert result.propellant.gamma == pytest.approx(1.21)
    assert result.propellant.c_star == pytest.approx(1750.0)
    assert result.report_sections["thermochemistry"]["source"] == "rocketcea"


def test_physics_trends_are_sensible():
    contour = bell_nozzle_contour(0.020, 8.0, length_pct=80.0)
    prop = get_propellant("LOX/RP-1")
    heat = bartz_heat_flux(contour, 80e5, prop)
    throat_x = contour["x"][abs(contour["y"] - contour["Rt"]).argmin()]

    assert abs(heat["x_q_max"] - throat_x) < 0.010

    bl = boundary_layer_displacement(contour, 80e5, prop)
    assert bl["effective_epsilon"] < contour["epsilon"]

    # Real coupled Sieder-Tate solve (gas side = full Bartz): more
    # coolant flow MUST lower the wall temperature (more cooling
    # capacity + faster coolant -> higher h_c).  A thin copper wall and
    # adequate flow keep the temperatures physical.
    material = MaterialSpec(conductivity=350.0, max_temperature=1300.0)

    def _cool(mdot):
        return regenerative_cooling_screen(
            heat, contour,
            CoolingSpec(method="regenerative", coolant="rp1",
                        channel_count=100, channel_width=0.0008,
                        channel_height=0.0025, coolant_mass_flow=mdot,
                        max_wall_temperature=1100.0),
            material, 0.001, prop, 80e5,
        )

    low_flow = _cool(8.0)
    high_flow = _cool(20.0)
    assert high_flow["model"] == "sieder_tate_1d_regen"
    assert (high_flow["estimated_wall_temperature"]
            < low_flow["estimated_wall_temperature"])
    # Coolant heats up along the channel (energy balance).
    assert high_flow["coolant_outlet_temperature"] > 293.0

    low_pressure = structural_screen(contour, 40e5, 101325, prop, material, 0.002, heat, high_flow)
    high_pressure = structural_screen(contour, 120e5, 101325, prop, material, 0.002, heat, high_flow)
    assert high_pressure["stress_margin"] < low_pressure["stress_margin"]
