from types import SimpleNamespace

import pytest

from raosim.spray_coupling import (
    SprayCStarCouplingSpec,
    solve_spray_cstar_fixed_point,
)
from raosim.spray.handoff import SprayCycleHandoff
from raosim.spray.solver import SprayGate


def test_fixed_point_closes_mass_flow_and_efficiency():
    # Synthetic geometry response: vaporization declines slightly as the mass
    # flow required by a lower eta increases.
    calls = []

    def evaluator(eta, mdot):
        calls.append((eta, mdot))
        eta_vap = 0.99 - 0.015 * (mdot - 2.0)
        return SimpleNamespace(eta_vaporization=eta_vap), {"mdot": mdot}

    spec = SprayCStarCouplingSpec(
        enabled=True,
        eta_mixing=0.98,
        eta_combustion=0.99,
        relaxation=0.6,
        relative_tolerance=1.0e-7,
    )
    result = solve_spray_cstar_fixed_point(
        spec,
        initial_eta_cstar=0.97,
        ideal_cstar=1000.0,
        chamber_pressure=2.0e6,
        throat_area=1.0e-3,
        evaluator=evaluator,
    )
    assert result.converged is True
    assert result.required_mass_flow == pytest.approx(
        2.0 / result.eta_cstar
    )
    assert result.payload["mdot"] == pytest.approx(result.required_mass_flow)
    assert len(calls) == len(result.iterations) + 1


def test_coupling_requires_explicit_mixing_and_combustion_efficiency():
    with pytest.raises(ValueError, match="eta_mixing"):
        SprayCStarCouplingSpec(enabled=True).validate()
    with pytest.raises(ValueError, match="source must"):
        SprayCStarCouplingSpec(source="unknown").validate()


def test_lagrangian_source_rejects_eta_duck_type_and_failed_handoff():
    spec = SprayCStarCouplingSpec(
        enabled=True,
        eta_mixing=0.98,
        eta_combustion=0.99,
        source="lagrangian",
    )
    common = dict(
        spec=spec,
        initial_eta_cstar=0.97,
        ideal_cstar=1500.0,
        chamber_pressure=3.0e6,
        throat_area=1.0e-3,
    )
    with pytest.raises(RuntimeError, match="typed SprayCycleHandoff"):
        solve_spray_cstar_fixed_point(
            **common,
            evaluator=lambda eta, mdot: (
                SimpleNamespace(eta_vaporization=0.9), None
            ),
        )

    handoff = SprayCycleHandoff(
        model_id="test",
        model_version="1",
        operating_point_id="point",
        smd_sampling_plane=0.01,
        streams=(),
        eta_vaporization=0.9,
        aggregation_basis="liquid_mass",
        conservation={},
        benchmark_evidence=(),
        carrier_provenance=(),
        solver_metadata={},
        convergence_evidence=None,
        required_gates=(SprayGate("strict_target_benchmark", "fail", "missing"),),
        fingerprint="0" * 64,
    )
    with pytest.raises(RuntimeError, match="strict_target_benchmark"):
        solve_spray_cstar_fixed_point(
            **common,
            evaluator=lambda eta, mdot: (handoff, None),
        )


def test_coupling_rejects_inapplicable_atomization():
    spec = SprayCStarCouplingSpec(
        enabled=True, eta_mixing=0.98, eta_combustion=0.99
    )
    with pytest.raises(RuntimeError, match="eta_vaporization"):
        solve_spray_cstar_fixed_point(
            spec,
            initial_eta_cstar=0.97,
            ideal_cstar=1500.0,
            chamber_pressure=3.0e6,
            throat_area=1.0e-3,
            evaluator=lambda eta, mdot: (
                SimpleNamespace(eta_vaporization=float("nan")), None
            ),
        )


def test_coupling_does_not_renormalize_over_an_inapplicable_stream():
    spec = SprayCStarCouplingSpec(
        enabled=True, eta_mixing=0.98, eta_combustion=0.99
    )
    atomization = SimpleNamespace(
        eta_vaporization=0.95,
        streams={
            "fuel": SimpleNamespace(applicable=False),
            "oxidizer": SimpleNamespace(applicable=True),
        },
    )
    with pytest.raises(RuntimeError, match="every propellant stream"):
        solve_spray_cstar_fixed_point(
            spec,
            initial_eta_cstar=0.97,
            ideal_cstar=1500.0,
            chamber_pressure=3.0e6,
            throat_area=1.0e-3,
            evaluator=lambda eta, mdot: (atomization, None),
        )


def test_coupling_rejects_liquid_stream_below_atomization_regime():
    spec = SprayCStarCouplingSpec(
        enabled=True, eta_mixing=0.98, eta_combustion=0.99
    )
    atomization = SimpleNamespace(
        eta_vaporization=0.95,
        streams={
            "fuel": SimpleNamespace(
                applicable=True,
                regime="below atomization regime (We_g=10 < 40)",
            ),
            "oxidizer": SimpleNamespace(
                applicable=True,
                regime="aerodynamic atomization",
            ),
        },
    )
    with pytest.raises(RuntimeError, match="atomization regime"):
        solve_spray_cstar_fixed_point(
            spec,
            initial_eta_cstar=0.97,
            ideal_cstar=1500.0,
            chamber_pressure=3.0e6,
            throat_area=1.0e-3,
            evaluator=lambda eta, mdot: (atomization, None),
        )


def test_coupling_enforces_minimum_cycle_efficiency():
    spec = SprayCStarCouplingSpec(
        enabled=True,
        eta_mixing=0.5,
        eta_combustion=0.5,
        minimum_eta_cstar=0.4,
    )
    with pytest.raises(RuntimeError, match="below configured minimum"):
        solve_spray_cstar_fixed_point(
            spec,
            initial_eta_cstar=0.9,
            ideal_cstar=1500.0,
            chamber_pressure=3.0e6,
            throat_area=1.0e-3,
            evaluator=lambda eta, mdot: (
                SimpleNamespace(eta_vaporization=0.9), None
            ),
        )
