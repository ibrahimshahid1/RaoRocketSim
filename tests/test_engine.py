"""
Tests for raosim.engine
"""

import math
import pytest
from raosim.engine import compute_engine_performance, g0
from raosim.propellants import custom_propellant


class TestEnginePerformance:

    @pytest.fixture
    def perf_air_like(self):
        """
        Compute performance for an 'air-like' gas to compare with the
        document's worked example: γ=1.4, R=287, Tc=3000, ε=10, Pc=10 MPa.
        """
        # Mw = R_universal / R_gas = 8314.46 / 287 ≈ 28.97 g/mol ≈ 0.02897 kg/mol
        prop = custom_propellant(gamma=1.4, Mw=0.02897, Tc=3000.0, eta_Isp=1.0)
        return compute_engine_performance(
            Pc=10e6, Pa=0.0, Rt=0.020, epsilon=10.0, prop=prop,
        )

    def test_exit_mach(self, perf_air_like):
        assert perf_air_like.Me == pytest.approx(3.9226, rel=1e-3)

    def test_cstar(self, perf_air_like):
        assert perf_air_like.c_star == pytest.approx(1355.0, rel=2e-2)

    def test_cf_vacuum(self, perf_air_like):
        """With Pa=0 and η=1, Cf_actual should equal Cf_ideal ≈ 1.647."""
        assert perf_air_like.Cf_actual == pytest.approx(1.647, rel=2e-2)

    def test_thrust_positive(self, perf_air_like):
        assert perf_air_like.thrust > 0

    def test_isp_vacuum(self, perf_air_like):
        """Isp_vac ≈ Cf * c* / g0 ≈ 1.647 * 1355 / 9.807 ≈ 228 s."""
        assert perf_air_like.Isp == pytest.approx(228.0, rel=3e-2)

    def test_mass_flow_rate(self, perf_air_like):
        """ṁ = Pc·At / c*."""
        At = math.pi * 0.020**2
        expected_mdot = 10e6 * At / perf_air_like.c_star
        assert perf_air_like.m_dot == pytest.approx(expected_mdot, rel=1e-3)


class TestEngineWithRealPropellant:
    def test_lox_rp1_runs(self):
        from raosim.propellants import get_propellant
        prop = get_propellant("LOX/RP-1")
        perf = compute_engine_performance(
            Pc=45e5, Pa=101325, Rt=0.020, epsilon=8.0, prop=prop,
        )
        assert perf.thrust > 0
        assert perf.Isp > 100
        assert perf.m_dot > 0
