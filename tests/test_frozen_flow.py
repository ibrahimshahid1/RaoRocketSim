"""Thermally-perfect frozen-composition quasi-1-D flow contracts."""

from __future__ import annotations

import hashlib
import json
import math
import sys

import pytest
import numpy as np

from raosim.frozen_flow import (
    MODEL_ID,
    FrozenFlowError,
    FrozenIdealGasTable,
    expansion_ratio_from_pressure_frozen,
    load_frozen_gas_table,
    solve_frozen_nozzle_expansion,
)
from raosim.gas_dynamics import (
    characteristic_velocity,
    isentropic_pressure_ratio,
    isentropic_temperature_ratio,
    mach_from_area_ratio,
    thrust_coefficient,
)
from raosim.engine import compute_engine_performance
from raosim.propellants import custom_propellant


def _gas_table(
    *,
    molecular_weight_kg_mol,
    composition_mass_fractions,
    temperature_nodes_k,
    cp_nodes_j_kg_k,
    source,
    freeze_basis="manufactured_composition",
    composition_state_pressure_pa=None,
    composition_state_temperature_k=None,
    mixture_ratio=None,
    generator="pytest_manufactured_oracle",
    generator_version="1",
    thermo_database="manufactured_analytic_cp",
    source_artifact_sha256=None,
    input_artifact_sha256=None,
):
    if source_artifact_sha256 is None:
        source_artifact_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return FrozenIdealGasTable(
        molecular_weight_kg_mol=molecular_weight_kg_mol,
        composition_mass_fractions=composition_mass_fractions,
        temperature_nodes_k=temperature_nodes_k,
        cp_nodes_j_kg_k=cp_nodes_j_kg_k,
        source=source,
        freeze_basis=freeze_basis,
        composition_state_pressure_pa=composition_state_pressure_pa,
        composition_state_temperature_k=composition_state_temperature_k,
        mixture_ratio=mixture_ratio,
        generator=generator,
        generator_version=generator_version,
        thermo_database=thermo_database,
        source_artifact_sha256=source_artifact_sha256,
        input_artifact_sha256=input_artifact_sha256,
    )


def _constant_gamma_gas(gamma: float = 1.4, gas_constant: float = 287.0):
    cp = gamma * gas_constant / (gamma - 1.0)
    return _gas_table(
        molecular_weight_kg_mol=8.31446261815324 / gas_constant,
        composition_mass_fractions={"manufactured_air": 1.0},
        temperature_nodes_k=(100.0, 4000.0),
        cp_nodes_j_kg_k=(cp, cp),
        source="manufactured constant-cp regression oracle",
    )


@pytest.mark.parametrize("gamma", (1.2, 1.3, 1.4, 1.667))
@pytest.mark.parametrize("epsilon", (1.0, 2.0, 10.0, 40.0))
def test_constant_cp_collapses_to_existing_constant_gamma_relations(gamma, epsilon):
    gas = _constant_gamma_gas(gamma)
    Pc = 10.0e6
    T0 = 3000.0
    Pa = 101325.0
    solved = solve_frozen_nozzle_expansion(
        gas,
        chamber_pressure_pa=Pc,
        chamber_temperature_k=T0,
        expansion_ratio=epsilon,
        ambient_pressure_pa=Pa,
    )
    # The legacy Newton inverse is ill-conditioned exactly at the double sonic
    # root and returns 1 +/- O(1e-8); the analytical value at A/A*=1 is exact.
    expected_mach = (
        1.0 if epsilon == 1.0
        else mach_from_area_ratio(epsilon, gamma, supersonic=True)
    )
    expected_t_ratio = isentropic_temperature_ratio(expected_mach, gamma)
    expected_p_ratio = isentropic_pressure_ratio(expected_mach, gamma)
    expected_cstar = characteristic_velocity(gamma, gas.gas_constant_j_kg_k, T0)
    expected_cf = thrust_coefficient(
        expected_mach, gamma, expected_p_ratio, Pa / Pc, epsilon
    )

    assert solved.exit.mach == pytest.approx(expected_mach, rel=2e-10)
    assert solved.exit.temperature_k / T0 == pytest.approx(
        expected_t_ratio, rel=2e-10
    )
    assert solved.exit.pressure_ratio == pytest.approx(expected_p_ratio, rel=3e-10)
    assert solved.characteristic_velocity_m_s == pytest.approx(
        expected_cstar, rel=2e-10
    )
    assert solved.thrust_coefficient == pytest.approx(expected_cf, rel=3e-10)
    assert solved.all_closures_pass


def test_throat_is_sonic_and_two_area_branches_are_distinct():
    solved = solve_frozen_nozzle_expansion(
        _constant_gamma_gas(),
        chamber_pressure_pa=5e6,
        chamber_temperature_k=3000.0,
        expansion_ratio=8.0,
    )
    subsonic = solved.station(3.0, supersonic=False)
    supersonic = solved.station(3.0, supersonic=True)

    assert solved.throat.mach == pytest.approx(1.0, rel=2e-10)
    assert subsonic.area_ratio == pytest.approx(3.0, rel=2e-10)
    assert supersonic.area_ratio == pytest.approx(3.0, rel=2e-10)
    assert 0.0 < subsonic.mach < 1.0 < supersonic.mach
    assert subsonic.temperature_k > solved.throat.temperature_k
    assert supersonic.temperature_k < solved.throat.temperature_k
    assert (
        subsonic.mass_flux_kg_m2_s * subsonic.area_ratio
        == pytest.approx(solved.throat.mass_flux_kg_m2_s, rel=2e-10)
    )
    assert (
        supersonic.mass_flux_kg_m2_s * supersonic.area_ratio
        == pytest.approx(solved.throat.mass_flux_kg_m2_s, rel=2e-10)
    )


def test_piecewise_linear_cp_integrals_are_exact_for_manufactured_line():
    # cp(T) = intercept + slope*T over the complete range.
    intercept = 700.0
    slope = 0.2
    gas = _gas_table(
        molecular_weight_kg_mol=0.028,
        composition_mass_fractions={"A": 0.75, "B": 0.25},
        temperature_nodes_k=(200.0, 1000.0, 2500.0, 4000.0),
        cp_nodes_j_kg_k=tuple(
            intercept + slope * value for value in (200.0, 1000.0, 2500.0, 4000.0)
        ),
        source="manufactured linear-cp exact-integral oracle",
    )
    t1, t2 = 350.0, 3375.0
    expected_h = intercept * (t2 - t1) + 0.5 * slope * (t2**2 - t1**2)
    expected_s = intercept * math.log(t2 / t1) + slope * (t2 - t1)
    assert gas.enthalpy_change(t1, t2) == pytest.approx(expected_h, rel=1e-13)
    assert gas.standard_entropy_change(t1, t2) == pytest.approx(
        expected_s, rel=1e-13
    )
    assert gas.enthalpy_change(t2, t1) == pytest.approx(-expected_h, rel=1e-13)
    assert gas.standard_entropy_change(t2, t1) == pytest.approx(
        -expected_s, rel=1e-13
    )


def test_variable_cp_solution_closes_energy_entropy_area_and_mass():
    gas = _gas_table(
        molecular_weight_kg_mol=0.022,
        composition_mass_fractions={"H2O": 0.4, "CO2": 0.35, "CO": 0.25},
        temperature_nodes_k=(200.0, 500.0, 1000.0, 2000.0, 3000.0, 3800.0),
        cp_nodes_j_kg_k=(1250.0, 1375.0, 1510.0, 1770.0, 1990.0, 2130.0),
        source="manufactured fixed-composition variable-cp table",
    )
    solved = solve_frozen_nozzle_expansion(
        gas,
        chamber_pressure_pa=8e6,
        chamber_temperature_k=3500.0,
        expansion_ratio=20.0,
        ambient_pressure_pa=20_000.0,
    )
    assert solved.throat.gamma != pytest.approx(solved.exit.gamma, rel=1e-3)
    assert solved.exit.temperature_k < solved.throat.temperature_k < 3500.0
    assert solved.sonic_relative_residual < 5e-10
    assert solved.exit_area_relative_residual < 5e-10
    assert solved.exit_mass_relative_residual < 5e-10
    assert solved.throat.energy_relative_residual < 5e-10
    assert solved.exit.entropy_relative_residual < 5e-10
    assert solved.thrust_coefficient > 0.0
    assert solved.characteristic_velocity_m_s > 0.0
    payload = solved.as_dict()
    assert payload["applicability"]["frozen_composition"] is True
    assert payload["applicability"]["equilibrium_chemistry"] is False
    assert payload["applicability"]["moc_or_rao_characteristics"] is False
    assert payload["applicability"]["profile_aware_wall_pressure_screen"] is True
    assert payload["applicability"]["empirical_separation_screen"] is True
    assert payload["applicability"]["separation_flow_solution"] is False


def test_pressure_scaling_leaves_dimensionless_state_and_cstar_unchanged():
    gas = _constant_gamma_gas(1.3, 350.0)
    low = solve_frozen_nozzle_expansion(
        gas,
        chamber_pressure_pa=1e6,
        chamber_temperature_k=3200.0,
        expansion_ratio=12.0,
    )
    high = solve_frozen_nozzle_expansion(
        gas,
        chamber_pressure_pa=9e6,
        chamber_temperature_k=3200.0,
        expansion_ratio=12.0,
    )
    assert high.exit.mach == pytest.approx(low.exit.mach, rel=1e-12)
    assert high.exit.pressure_ratio == pytest.approx(low.exit.pressure_ratio, rel=1e-12)
    assert high.characteristic_velocity_m_s == pytest.approx(
        low.characteristic_velocity_m_s, rel=1e-12
    )
    assert high.exit.pressure_pa == pytest.approx(9.0 * low.exit.pressure_pa)


def test_matched_pressure_inverse_roundtrips_constant_and_variable_cp():
    for gas in (
        _constant_gamma_gas(1.3, 350.0),
        _gas_table(
            molecular_weight_kg_mol=0.022,
            composition_mass_fractions={"products": 1.0},
            temperature_nodes_k=(200.0, 800.0, 1600.0, 2400.0, 3200.0, 3800.0),
            cp_nodes_j_kg_k=(1250.0, 1450.0, 1680.0, 1870.0, 2030.0, 2120.0),
            source="manufactured pressure inverse",
        ),
    ):
        original = solve_frozen_nozzle_expansion(
            gas,
            chamber_pressure_pa=8e6,
            chamber_temperature_k=3500.0,
            expansion_ratio=9.0,
        )
        epsilon, station = expansion_ratio_from_pressure_frozen(
            gas,
            chamber_pressure_pa=8e6,
            chamber_temperature_k=3500.0,
            exit_pressure_pa=original.exit.pressure_pa,
        )
        assert epsilon == pytest.approx(9.0, rel=3e-10)
        assert station.pressure_pa == pytest.approx(original.exit.pressure_pa, rel=3e-10)
        assert station.mach == pytest.approx(original.exit.mach, rel=3e-10)


def test_engine_dispatch_constant_cp_matches_legacy_performance():
    gamma = 1.3
    gas_constant = 350.0
    gas = _constant_gamma_gas(gamma, gas_constant)
    prop = custom_propellant(
        gamma=gamma,
        Mw=gas.molecular_weight_kg_mol,
        Tc=3200.0,
        eta_cstar=0.97,
        eta_CF=0.98,
    )
    legacy = compute_engine_performance(
        Pc=7e6, Pa=50_000.0, Rt=0.025, epsilon=12.0, prop=prop
    )
    frozen = compute_engine_performance(
        Pc=7e6,
        Pa=50_000.0,
        Rt=0.025,
        epsilon=12.0,
        prop=prop,
        frozen_gas=gas,
    )
    assert frozen.expansion_model == "frozen_variable_cp_q1d"
    assert frozen.Me == pytest.approx(legacy.Me, rel=3e-10)
    assert frozen.Pe == pytest.approx(legacy.Pe, rel=3e-10)
    assert frozen.Cf_ideal == pytest.approx(legacy.Cf_ideal, rel=3e-10)
    # Propellant keeps the repository's historical rounded universal gas
    # constant (8314.46 J/kmol/K); the new table uses the CODATA value.
    assert frozen.c_star == pytest.approx(legacy.c_star, rel=3e-7)
    assert frozen.m_dot == pytest.approx(legacy.m_dot, rel=3e-7)
    assert frozen.thrust == pytest.approx(legacy.thrust, rel=3e-10)
    assert frozen.gamma_throat == pytest.approx(gamma)
    assert frozen.gamma_exit == pytest.approx(gamma)
    assert frozen.frozen_flow is not None
    assert frozen.frozen_flow_fingerprint == frozen.frozen_flow.input_fingerprint_sha256


def test_engine_variable_cp_uses_integrated_cstar_and_rejects_mixed_composition():
    gas = _gas_table(
        molecular_weight_kg_mol=0.022,
        composition_mass_fractions={"products": 1.0},
        temperature_nodes_k=(200.0, 1000.0, 2000.0, 3000.0, 3800.0),
        cp_nodes_j_kg_k=(1250.0, 1500.0, 1750.0, 1980.0, 2100.0),
        source="manufactured products",
    )
    prop = custom_propellant(
        gamma=1.22, Mw=0.022, Tc=3500.0, eta_cstar=0.98, eta_CF=0.99
    )
    performance = compute_engine_performance(
        Pc=8e6,
        Pa=101325.0,
        Rt=0.02,
        epsilon=10.0,
        prop=prop,
        frozen_gas=gas,
    )
    assert performance.c_star != pytest.approx(prop.c_star, rel=1e-4)
    assert performance.reference_propellant_c_star == pytest.approx(prop.c_star)
    assert performance.gamma_throat != pytest.approx(performance.gamma_exit, rel=1e-3)
    assert performance.exit_temperature == pytest.approx(
        performance.frozen_flow.exit.temperature_k
    )

    wrong = _constant_gamma_gas()
    with pytest.raises(ValueError, match="molecular weight does not match"):
        compute_engine_performance(
            Pc=8e6,
            Pa=0.0,
            Rt=0.02,
            epsilon=10.0,
            prop=prop,
            frozen_gas=wrong,
        )


def test_wall_pressure_and_separation_consume_frozen_profile_without_legacy_inverse(
    monkeypatch,
):
    gas = _gas_table(
        molecular_weight_kg_mol=0.022,
        composition_mass_fractions={"products": 1.0},
        temperature_nodes_k=(200.0, 800.0, 1600.0, 2400.0, 3200.0, 3800.0),
        cp_nodes_j_kg_k=(1250.0, 1450.0, 1680.0, 1870.0, 2030.0, 2120.0),
        source="manufactured profile adapter test",
    )
    expansion = solve_frozen_nozzle_expansion(
        gas,
        chamber_pressure_pa=8e6,
        chamber_temperature_k=3500.0,
        expansion_ratio=10.0,
        ambient_pressure_pa=101325.0,
    )
    Rt = 0.02
    area_ratios = np.array([1.0, 1.2, 2.0, 4.0, 7.0, 10.0])
    contour = {
        "x": np.linspace(0.0, 0.2, area_ratios.size),
        "y": Rt * np.sqrt(area_ratios),
        "Rt": Rt,
        "epsilon": 10.0,
    }
    import raosim.wall_pressure as wall_pressure_module
    import raosim.separation as separation_module

    def forbidden(*_args, **_kwargs):
        raise AssertionError("constant-gamma area-Mach inverse must not run")

    monkeypatch.setattr(wall_pressure_module, "mach_from_area_ratio", forbidden)
    monkeypatch.setattr(separation_module, "mach_from_area_ratio", forbidden)
    wall = wall_pressure_module.wall_pressure_distribution(
        contour, 8e6, 1.22, frozen_expansion=expansion
    )
    separation = separation_module.check_separation(
        contour,
        8e6,
        101325.0,
        1.22,
        frozen_expansion=expansion,
    )
    assert wall["expansion_model"] == "frozen_variable_cp_q1d"
    assert wall["monotonic"]
    assert np.all(np.diff(wall["p"]) < 0.0)
    assert wall["M"][-1] == pytest.approx(expansion.exit.mach)
    assert wall["T"][-1] == pytest.approx(expansion.exit.temperature_k)
    assert wall["gamma"][0] != pytest.approx(wall["gamma"][-1], rel=1e-3)
    assert separation["expansion_model"] == "frozen_variable_cp_q1d"
    assert separation["exit_pressure"] == pytest.approx(expansion.exit.pressure_pa)


def test_profile_adapters_reject_stale_pressure_or_expansion_ratio():
    gas = _constant_gamma_gas()
    expansion = solve_frozen_nozzle_expansion(
        gas,
        chamber_pressure_pa=8e6,
        chamber_temperature_k=3000.0,
        expansion_ratio=5.0,
    )
    contour = {
        "x": np.array([0.0, 0.1]),
        "y": np.array([0.02, 0.02 * math.sqrt(5.0)]),
        "Rt": 0.02,
        "epsilon": 5.0,
    }
    from raosim.wall_pressure import wall_pressure_distribution
    from raosim.separation import check_separation

    with pytest.raises(ValueError, match="chamber pressure"):
        wall_pressure_distribution(
            contour, 7e6, 1.4, frozen_expansion=expansion
        )
    wrong_contour = {**contour, "epsilon": 6.0}
    with pytest.raises(ValueError, match="expansion ratio"):
        check_separation(
            wrong_contour, 8e6, 0.0, 1.4, frozen_expansion=expansion
        )


def test_fingerprint_is_deterministic_and_changes_with_property_evidence():
    first = _constant_gamma_gas()
    second = _constant_gamma_gas()
    changed = _gas_table(
        molecular_weight_kg_mol=first.molecular_weight_kg_mol,
        composition_mass_fractions=first.composition_mass_fractions,
        temperature_nodes_k=first.temperature_nodes_k,
        cp_nodes_j_kg_k=(first.cp_nodes_j_kg_k[0], first.cp_nodes_j_kg_k[1] + 1.0),
        source=first.source,
    )
    assert first.fingerprint_sha256 == second.fingerprint_sha256
    assert first.fingerprint_sha256 != changed.fingerprint_sha256

    changed_provenance = _gas_table(
        molecular_weight_kg_mol=first.molecular_weight_kg_mol,
        composition_mass_fractions=first.composition_mass_fractions,
        temperature_nodes_k=first.temperature_nodes_k,
        cp_nodes_j_kg_k=first.cp_nodes_j_kg_k,
        source=first.source,
        generator_version="2",
    )
    assert first.fingerprint_sha256 != changed_provenance.fingerprint_sha256
    payload = first.as_dict()
    assert payload["schema"] == "raosim.frozen_ideal_gas_table.v2"
    assert payload["freeze_basis"] == "manufactured_composition"
    assert payload["generator"] == "pytest_manufactured_oracle"
    assert payload["source_artifact_sha256"]


def test_chamber_equilibrium_snapshot_is_bound_to_generation_pressure_temperature():
    gas = _gas_table(
        molecular_weight_kg_mol=0.022,
        composition_mass_fractions={"products": 1.0},
        temperature_nodes_k=(200.0, 1000.0, 2000.0, 3000.0, 3800.0),
        cp_nodes_j_kg_k=(1250.0, 1500.0, 1750.0, 1980.0, 2100.0),
        source="manufactured chamber-equilibrium snapshot contract",
        freeze_basis="chamber_equilibrium_snapshot",
        composition_state_pressure_pa=8.0e6,
        composition_state_temperature_k=3500.0,
        mixture_ratio=2.6,
        generator="CEA-compatible test fixture",
        generator_version="fixture-1",
        thermo_database="manufactured fixture database",
    )
    solved = solve_frozen_nozzle_expansion(
        gas,
        chamber_pressure_pa=8.0e6,
        chamber_temperature_k=3500.0,
        expansion_ratio=4.0,
    )
    assert solved.gas.mixture_ratio == pytest.approx(2.6)

    with pytest.raises(FrozenFlowError, match="snapshot pressure"):
        solve_frozen_nozzle_expansion(
            gas,
            chamber_pressure_pa=7.9e6,
            chamber_temperature_k=3500.0,
            expansion_ratio=4.0,
        )
    with pytest.raises(FrozenFlowError, match="snapshot temperature"):
        solve_frozen_nozzle_expansion(
            gas,
            chamber_pressure_pa=8.0e6,
            chamber_temperature_k=3490.0,
            expansion_ratio=4.0,
        )


def test_externally_fixed_composition_is_not_bound_to_reference_pressure():
    gas = _gas_table(
        molecular_weight_kg_mol=0.02897,
        composition_mass_fractions={"air": 1.0},
        temperature_nodes_k=(200.0, 4000.0),
        cp_nodes_j_kg_k=(1005.0, 1005.0),
        source="externally fixed reference composition",
        freeze_basis="externally_fixed_composition",
        composition_state_pressure_pa=101325.0,
        composition_state_temperature_k=300.0,
    )
    for pressure in (1.0e6, 9.0e6):
        solved = solve_frozen_nozzle_expansion(
            gas,
            chamber_pressure_pa=pressure,
            chamber_temperature_k=3000.0,
            expansion_ratio=3.0,
        )
        assert solved.chamber_pressure_pa == pressure


def test_strict_json_loader_binds_exact_file_hash(tmp_path):
    source = "configuration-controlled test property table"
    payload = {
        "schema_version": 2,
        "model": MODEL_ID,
        "molecular_weight_kg_mol": 0.02897,
        "composition_mass_fractions": {"N2": 0.77, "O2": 0.23},
        "temperature_nodes_k": [200.0, 1000.0, 4000.0],
        "cp_nodes_j_kg_k": [1000.0, 1100.0, 1300.0],
        "source": source,
        "freeze_basis": "externally_fixed_composition",
        "composition_state_pressure_pa": None,
        "composition_state_temperature_k": None,
        "mixture_ratio": None,
        "generator": "pytest",
        "generator_version": "1",
        "thermo_database": "manufactured",
        "source_artifact_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }
    path = tmp_path / "gas.json"
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    path.write_bytes(raw)
    gas = load_frozen_gas_table(path)
    assert gas.input_artifact_sha256 == hashlib.sha256(raw).hexdigest()
    assert gas.composition_mass_fractions == {"N2": 0.77, "O2": 0.23}
    assert gas.fingerprint_sha256


@pytest.mark.parametrize(
    "builder,match",
    [
        (
            lambda: _gas_table(
                molecular_weight_kg_mol=0.028,
                composition_mass_fractions={"air": 1.0},
                temperature_nodes_k=(200.0, 4000.0),
                cp_nodes_j_kg_k=(200.0, 200.0),
                source="bad cp",
            ),
            "cv > 0",
        ),
        (
            lambda: _gas_table(
                molecular_weight_kg_mol=0.028,
                composition_mass_fractions={"air": 0.9},
                temperature_nodes_k=(200.0, 4000.0),
                cp_nodes_j_kg_k=(1000.0, 1000.0),
                source="bad sum",
            ),
            "sum to one",
        ),
        (
            lambda: _gas_table(
                molecular_weight_kg_mol=0.028,
                composition_mass_fractions={"air": 1.0},
                temperature_nodes_k=(4000.0, 200.0),
                cp_nodes_j_kg_k=(1000.0, 1000.0),
                source="bad T",
            ),
            "strictly increasing",
        ),
        (
            lambda: _gas_table(
                molecular_weight_kg_mol=0.028,
                composition_mass_fractions={"air": 1.0},
                temperature_nodes_k=(200.0, 4000.0),
                cp_nodes_j_kg_k=(math.nan, 1000.0),
                source="nan",
            ),
            "finite",
        ),
    ],
)
def test_invalid_property_tables_fail_closed(builder, match):
    with pytest.raises(FrozenFlowError, match=match):
        builder()


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"freeze_basis": "unknown"}, "freeze_basis must be one of"),
        (
            {
                "freeze_basis": "chamber_equilibrium_snapshot",
                "composition_state_pressure_pa": None,
                "composition_state_temperature_k": None,
                "mixture_ratio": None,
            },
            "chamber_equilibrium_snapshot requires",
        ),
        ({"source_artifact_sha256": "not-a-hash"}, "source_artifact_sha256"),
        (
            {"composition_state_temperature_k": 5000.0},
            "inside the cp table bounds",
        ),
    ],
)
def test_invalid_property_provenance_fails_closed(overrides, match):
    arguments = {
        "molecular_weight_kg_mol": 0.02897,
        "composition_mass_fractions": {"air": 1.0},
        "temperature_nodes_k": (200.0, 4000.0),
        "cp_nodes_j_kg_k": (1005.0, 1005.0),
        "source": "provenance failure fixture",
    }
    arguments.update(overrides)
    with pytest.raises(FrozenFlowError, match=match):
        _gas_table(**arguments)


@pytest.mark.parametrize("chamber_pressure", (math.ulp(0.0), sys.float_info.min))
def test_zero_and_subnormal_mass_fluxes_fail_before_conservation_division(
    chamber_pressure,
):
    with pytest.raises(FrozenFlowError, match="mass flux.*underflow"):
        solve_frozen_nozzle_expansion(
            _constant_gamma_gas(),
            chamber_pressure_pa=chamber_pressure,
            chamber_temperature_k=3000.0,
            expansion_ratio=2.0,
        )


def test_static_pressure_ratio_underflow_fails_closed():
    gas = _gas_table(
        molecular_weight_kg_mol=0.02897,
        composition_mass_fractions={"manufactured": 1.0},
        temperature_nodes_k=(100.0, 4000.0),
        cp_nodes_j_kg_k=(1.0e8, 1.0e8),
        source="deliberate pressure-ratio underflow fixture",
    )
    with pytest.raises(FrozenFlowError, match="pressure ratio.*underflow"):
        solve_frozen_nozzle_expansion(
            gas,
            chamber_pressure_pa=1.0e6,
            chamber_temperature_k=3000.0,
            expansion_ratio=2.0,
        )


def test_no_extrapolation_and_insufficient_low_temperature_coverage_fail():
    gas = _gas_table(
        molecular_weight_kg_mol=0.02897,
        composition_mass_fractions={"air": 1.0},
        temperature_nodes_k=(2600.0, 3000.0, 3500.0),
        cp_nodes_j_kg_k=(1005.0, 1005.0, 1005.0),
        source="deliberately too-narrow table",
    )
    with pytest.raises(FrozenFlowError, match="low enough temperature"):
        solve_frozen_nozzle_expansion(
            gas,
            chamber_pressure_pa=1e6,
            chamber_temperature_k=3000.0,
            expansion_ratio=2.0,
        )
    with pytest.raises(FrozenFlowError, match="outside cp table bounds"):
        gas.cp(2000.0)


def test_loader_rejects_unknown_keys_and_symlinks(tmp_path):
    source = "test"
    payload = {
        "schema_version": 2,
        "model": MODEL_ID,
        "molecular_weight_kg_mol": 0.02897,
        "composition_mass_fractions": {"air": 1.0},
        "temperature_nodes_k": [200.0, 4000.0],
        "cp_nodes_j_kg_k": [1005.0, 1005.0],
        "source": source,
        "freeze_basis": "manufactured_composition",
        "composition_state_pressure_pa": None,
        "composition_state_temperature_k": None,
        "mixture_ratio": None,
        "generator": "pytest",
        "generator_version": "1",
        "thermo_database": "manufactured",
        "source_artifact_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "unexpected": True,
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FrozenFlowError, match="keys must exactly equal"):
        load_frozen_gas_table(path)
    link = tmp_path / "linked.json"
    link.symlink_to(path)
    with pytest.raises(FrozenFlowError, match="normal JSON file"):
        load_frozen_gas_table(link)


def test_strict_json_loader_rejects_duplicate_nonobject_and_wrong_types(tmp_path):
    source = "strict loader fixture"
    payload = {
        "schema_version": 2,
        "model": MODEL_ID,
        "molecular_weight_kg_mol": 0.02897,
        "composition_mass_fractions": {"air": 1.0},
        "temperature_nodes_k": [200.0, 4000.0],
        "cp_nodes_j_kg_k": [1005.0, 1005.0],
        "source": source,
        "freeze_basis": "manufactured_composition",
        "composition_state_pressure_pa": None,
        "composition_state_temperature_k": None,
        "mixture_ratio": None,
        "generator": "pytest",
        "generator_version": "1",
        "thermo_database": "manufactured",
        "source_artifact_sha256": hashlib.sha256(source.encode()).hexdigest(),
    }
    path = tmp_path / "strict.json"

    path.write_text("[]", encoding="utf-8")
    with pytest.raises(FrozenFlowError, match="root must be an object"):
        load_frozen_gas_table(path)

    path.write_text('{"schema_version":2,"schema_version":2}', encoding="utf-8")
    with pytest.raises(FrozenFlowError, match="duplicate JSON key"):
        load_frozen_gas_table(path)

    wrong_array = {**payload, "temperature_nodes_k": "200, 4000"}
    path.write_text(json.dumps(wrong_array), encoding="utf-8")
    with pytest.raises(FrozenFlowError, match="temperature_nodes_k must be an array"):
        load_frozen_gas_table(path)

    wrong_number = {**payload, "molecular_weight_kg_mol": "0.02897"}
    path.write_text(json.dumps(wrong_number), encoding="utf-8")
    with pytest.raises(FrozenFlowError, match="must be a JSON number"):
        load_frozen_gas_table(path)
