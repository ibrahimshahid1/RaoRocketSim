"""
Tests for raosim.gas_dynamics

Validated against the worked example in the Rao nozzle formulation document:
  ε = 10, γ = 1.4  →  Me ≈ 3.9226, pe/p0 ≈ 0.007306, Te/T0 ≈ 0.2453
                       ν(Me) ≈ 64.75°,  Cf_vac ≈ 1.647
                       c*(γ=1.4, R=287, T=3000) ≈ 1355 m/s
"""

import math
import pytest
from raosim.gas_dynamics import (
    area_mach_relation,
    mach_from_area_ratio,
    isentropic_pressure_ratio,
    isentropic_temperature_ratio,
    prandtl_meyer,
    thrust_coefficient,
    characteristic_velocity,
    expansion_ratio_from_pressure,
)


class TestAreaMachRelation:
    def test_sonic_throat(self):
        """A/A* = 1 at M = 1 for any γ."""
        assert area_mach_relation(1.0, 1.4) == pytest.approx(1.0, abs=1e-12)
        assert area_mach_relation(1.0, 1.2) == pytest.approx(1.0, abs=1e-12)

    def test_known_area_ratio(self):
        """ε = 10 at M ≈ 3.9226 for γ = 1.4."""
        M = 3.9226
        ar = area_mach_relation(M, 1.4)
        assert ar == pytest.approx(10.0, rel=1e-3)


class TestMachFromAreaRatio:
    def test_supersonic_root_eps10(self):
        """M(ε=10, γ=1.4) ≈ 3.9226."""
        M = mach_from_area_ratio(10.0, 1.4, supersonic=True)
        assert M == pytest.approx(3.9226, rel=1e-3)

    def test_roundtrip(self):
        """A/A*(M(A/A*)) should give back the original area ratio."""
        for eps in [2.0, 5.0, 10.0, 25.0, 50.0]:
            M = mach_from_area_ratio(eps, 1.4)
            assert area_mach_relation(M, 1.4) == pytest.approx(eps, rel=1e-6)


class TestIsentropicRatios:
    """Verified against the document: γ=1.4, Me=3.9226."""

    def test_pressure_ratio(self):
        pr = isentropic_pressure_ratio(3.9226, 1.4)
        assert pr == pytest.approx(0.007306, rel=2e-2)

    def test_temperature_ratio(self):
        tr = isentropic_temperature_ratio(3.9226, 1.4)
        assert tr == pytest.approx(0.2453, rel=1e-2)


class TestPrandtlMeyer:
    def test_nu_at_mach_1(self):
        """ν(1) = 0 for any γ."""
        assert prandtl_meyer(1.0, 1.4) == pytest.approx(0.0, abs=1e-12)

    def test_nu_at_me(self):
        """ν(3.9226, γ=1.4) ≈ 64.75°."""
        nu_deg = math.degrees(prandtl_meyer(3.9226, 1.4))
        assert nu_deg == pytest.approx(64.75, rel=1e-2)


class TestThrustCoefficient:
    def test_cf_vacuum(self):
        """Cf_vac at ε=10, γ=1.4 ≈ 1.647."""
        Me = mach_from_area_ratio(10.0, 1.4)
        Pe_Pc = isentropic_pressure_ratio(Me, 1.4)
        cf = thrust_coefficient(Me, 1.4, Pe_Pc, 0.0, 10.0)
        assert cf == pytest.approx(1.647, rel=2e-2)

    def test_cf_sea_level(self):
        """Cf at sea level with Pc=10 MPa, ε=10, γ=1.4 ≈ 1.546."""
        Me = mach_from_area_ratio(10.0, 1.4)
        Pe_Pc = isentropic_pressure_ratio(Me, 1.4)
        Pa_Pc = 101325 / 10e6
        cf = thrust_coefficient(Me, 1.4, Pe_Pc, Pa_Pc, 10.0)
        assert cf == pytest.approx(1.546, rel=2e-2)


class TestCharacteristicVelocity:
    def test_cstar_air(self):
        """c*(γ=1.4, R=287, T=3000) ≈ 1355 m/s."""
        cstar = characteristic_velocity(1.4, 287.0, 3000.0)
        assert cstar == pytest.approx(1355.0, rel=2e-2)


class TestExpansionRatioFromPressure:
    def test_roundtrip(self):
        """eps_from_pressure should be consistent with isentropic ratios."""
        Pc = 10e6
        Pa = 101325
        eps, Me = expansion_ratio_from_pressure(Pc, Pa, 1.4)
        # Verify Me gives the right pressure ratio
        Pe_Pc = isentropic_pressure_ratio(Me, 1.4)
        assert Pe_Pc == pytest.approx(Pa / Pc, rel=1e-4)
