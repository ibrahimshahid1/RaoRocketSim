"""Low-cycle fatigue: Coffin-Manson N_f model + its gating policy.

The nominal strain driver is α·ΔT, obtained directly from SP-125 equation
4-28's thermal stress E·α·ΔT; N_f inverts the standard strain-life relation.
The local literature (``propulsion_texts``)
carries no sourced alloy LCF coefficients, so the catalog supplies none:
fatigue is only *evaluated* with explicit sourced coefficients, and only
*gates* feasibility when those data are marked design-qualified.
"""
from __future__ import annotations

import pytest

from raosim.design import MaterialSpec
from raosim.nozzle_geometry import bell_nozzle_contour
from raosim.physics import coffin_manson_cycles, thermal_fatigue_strain
from raosim.propellants import custom_propellant
from raosim.thermal_design import joint_wall_channel_design


# --------------------------------------------------------------------- #
#  Coffin-Manson strain-life (pure physics — independent of the catalog)
# --------------------------------------------------------------------- #
def _cm(**over):
    kw = dict(elastic_modulus=140e9, fatigue_strength_coeff=350e6,
              fatigue_strength_exp=-0.09, fatigue_ductility_coeff=0.35,
              fatigue_ductility_exp=-0.6)
    kw.update(over)
    return kw


def test_coffin_manson_monotonic_in_strain():
    kw = _cm()
    cyc = [coffin_manson_cycles(de, **kw)
           for de in (0.001, 0.002, 0.004, 0.008, 0.016)]
    assert all(cyc[i] > cyc[i + 1] for i in range(len(cyc) - 1))


def test_coffin_manson_satisfies_the_strain_life_equation():
    """The returned N_f must satisfy Δε/2 = (σf'/E)(2Nf)^b + εf'(2Nf)^c."""
    kw = _cm(); de = 0.006
    Nf = coffin_manson_cycles(de, **kw)
    two_N = 2.0 * Nf
    rhs = ((kw["fatigue_strength_coeff"] / kw["elastic_modulus"])
           * two_N ** kw["fatigue_strength_exp"]
           + kw["fatigue_ductility_coeff"] * two_N ** kw["fatigue_ductility_exp"])
    assert rhs == pytest.approx(de / 2.0, rel=1e-3)


def test_coffin_manson_runout_and_overstrain_bounds():
    kw = _cm()
    assert coffin_manson_cycles(1e-9, **kw) >= 1e8     # ~zero strain → run-out
    assert coffin_manson_cycles(1.0, **kw) <= 1.0      # huge strain → ≤ half cycle


def test_tougher_alloy_outlasts_copper_at_equal_strain():
    """An Inconel-class coefficient set far outlasts a copper-class one at
    the same strain — the conductivity-versus-fatigue trade the catalog
    embodies."""
    copper = coffin_manson_cycles(0.006, **_cm())
    inconel = coffin_manson_cycles(0.006, **_cm(
        elastic_modulus=200e9, fatigue_strength_coeff=1800e6,
        fatigue_strength_exp=-0.07, fatigue_ductility_coeff=1.2,
        fatigue_ductility_exp=-0.7))
    assert inconel > 100 * copper


def test_thermal_fatigue_strain_driver():
    s1 = thermal_fatigue_strain(100.0, thermal_expansion=16.5e-6, poisson_ratio=0.33)
    s2 = thermal_fatigue_strain(200.0, thermal_expansion=16.5e-6, poisson_ratio=0.33)
    assert s2 == pytest.approx(2.0 * s1, rel=1e-9)         # linear in ΔT
    # SP-125 eq. 4-28 gives S_l=EαΔT, so its nominal elastic strain scale
    # is αΔT.  A biaxial 1/(1-v) multiplier must be an explicit assumption.
    assert s1 == pytest.approx(16.5e-6 * 100.0, rel=1e-9)
    s3 = thermal_fatigue_strain(100.0, thermal_expansion=16.5e-6,
                                poisson_ratio=0.33, mechanical_strain=1e-3)
    assert s3 == pytest.approx(s1 + 1e-3, rel=1e-9)        # mechanical adds


# --------------------------------------------------------------------- #
#  Gating policy in the joint optimizer (the P1 "generic data must not
#  gate feasibility" fix)
# --------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def prop():
    return custom_propellant(gamma=1.24, Mw=0.022, Tc=3500.0)


@pytest.fixture(scope="module")
def big_contour():
    return bell_nozzle_contour(Rt=0.08, epsilon=10.0, gamma=1.24, length_pct=80.0)


def _grcop_with_fatigue(*, qualified: bool) -> MaterialSpec:
    """GRCop-84 plus an explicitly *sourced* Coffin-Manson coefficient set
    (copper-class), optionally flagged design-qualified."""
    m = MaterialSpec.from_catalog("grcop-84")
    m.fatigue_strength_coeff = 350e6
    m.fatigue_strength_exp = -0.09
    m.fatigue_ductility_coeff = 0.35
    m.fatigue_ductility_exp = -0.6
    m.fatigue_source = "test LCF handbook (synthetic)"
    m.fatigue_design_qualified = qualified
    return m


_GRID = dict(mixture_ratio=2.0, thermal_margin=1.05, structural_fos=1.0,
             channel_height=0.005, t_hot_max=0.0025,
             n_t=5, n_w=3, n_count=4, n_iter=10)


def test_catalog_material_does_not_evaluate_fatigue(prop, big_contour):
    """A plain catalog material has no sourced LCF data, so N_f is not
    evaluated and cannot gate — generic numbers never decide feasibility."""
    r = joint_wall_channel_design(big_contour, prop, 1.0e6,
                                  material="grcop-84", **_GRID)
    assert r["fatigue_status"] == "not_evaluated_missing_sourced_coefficients"
    assert r["fatigue_cycles"] is None
    assert r["fatigue_gates_feasibility"] is False


def test_sourced_but_unqualified_fatigue_is_screening_only(prop, big_contour):
    """Sourced coefficients that are NOT design-qualified report an N_f for
    information but never force infeasibility — even an absurd cycle
    requirement leaves feasibility to thermal/structural/Δp."""
    m = _grcop_with_fatigue(qualified=False)
    r = joint_wall_channel_design(big_contour, prop, 1.0e6, material=m,
                                  required_cycles=1e9, life_fos=1.0, **_GRID)
    assert r["fatigue_status"] == "screening_only_not_gating"
    assert r["fatigue_gates_feasibility"] is False
    assert r["fatigue_cycles"] is not None        # reported for information
    assert r["feasible"] is True                  # but did NOT gate


def test_design_qualified_fatigue_gates_feasibility(prop, big_contour):
    """Design-qualified data DO gate: an unreachable cycle requirement makes
    the design infeasible and names the fatigue binder, while a lenient one
    passes."""
    m = _grcop_with_fatigue(qualified=True)
    strict = joint_wall_channel_design(big_contour, prop, 1.0e6, material=m,
                                       required_cycles=1e9, life_fos=1.0, **_GRID)
    assert strict["fatigue_status"] == "design_qualified_gate"
    assert strict["fatigue_gates_feasibility"] is True
    assert strict["feasible"] is False
    assert "fatigue" in strict["diagnosis"]

    lenient = joint_wall_channel_design(big_contour, prop, 1.0e6, material=m,
                                        required_cycles=10, life_fos=1.0, **_GRID)
    assert lenient["feasible"] is True
    assert lenient["fatigue_cycles"] is not None
