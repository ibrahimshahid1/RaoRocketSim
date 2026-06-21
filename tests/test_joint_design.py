"""Joint wall + channel optimizer (raosim.thermal_design).

Co-sizes the hot-wall thickness AND the channels against the coupled
thermal limit (peak wall T ≤ material T_max / margin) and the SP-125
eq. 4-31 combined structural stress, reporting the thickness feasibility
band  max(t_pressure, t_mfg) ≤ t_hot ≤ t_thermal.
"""
from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest

from raosim.nozzle_geometry import bell_nozzle_contour
from raosim.propellants import custom_propellant
from raosim.thermal_design import (
    _resolve_material, joint_wall_channel_design, wall_feasibility_band,
)


@pytest.fixture(scope="module")
def prop():
    return custom_propellant(gamma=1.24, Mw=0.022, Tc=3500.0)


@pytest.fixture(scope="module")
def big_contour():
    # 80 mm throat: large enough that its own fuel can cool it.
    return bell_nozzle_contour(Rt=0.08, epsilon=10.0, gamma=1.24, length_pct=80.0)


# --------------------------------------------------------------------- #
#  The SP-125 thickness feasibility band (pure, no solve)
# --------------------------------------------------------------------- #
def test_wall_feasibility_band_squeeze():
    """Thermal caps t_hot from above (thicker ⇒ hotter wall); structural
    brackets it (pressure thin-side, thermal-stress thick-side); the
    manufacturing floor lifts the lower bound."""
    t = [0.5e-3, 1.0e-3, 1.5e-3, 2.0e-3, 2.5e-3]
    thermal = [1.4, 1.2, 1.05, 0.90, 0.80]      # falls with thickness
    structural = [0.90, 1.10, 1.20, 1.05, 0.85]  # interior peak
    band = wall_feasibility_band(t, thermal, structural,
                                 thermal_target=1.0, structural_fos=1.0,
                                 t_mfg=0.6e-3)
    assert band["t_thermal_max"] == pytest.approx(1.5e-3)
    assert band["t_structural_lo"] == pytest.approx(1.0e-3)
    assert band["t_structural_hi"] == pytest.approx(2.0e-3)
    assert band["feasible_lo"] == pytest.approx(1.0e-3)   # max(mfg, struct_lo)
    assert band["feasible_hi"] == pytest.approx(1.5e-3)   # min(thermal, struct_hi)
    assert band["feasible"] is True


def test_wall_feasibility_band_empty_when_thermal_and_structural_disjoint():
    t = [0.5e-3, 1.0e-3, 1.5e-3]
    thermal = [1.2, 0.9, 0.8]        # only the thinnest is cool enough
    structural = [0.7, 0.8, 1.1]     # only the thickest is strong enough
    band = wall_feasibility_band(t, thermal, structural,
                                 thermal_target=1.0, structural_fos=1.0,
                                 t_mfg=0.5e-3)
    assert band["feasible"] is False


def test_resolve_material_requires_structural_properties():
    # A bare conductivity-only namespace can't supply E/α/ν for eq. 4-31.
    with pytest.raises(ValueError):
        _resolve_material(SimpleNamespace(conductivity=350.0))
    # A catalog name resolves to a fully-populated material.
    m = _resolve_material("grcop-84")
    assert m.elastic_modulus and m.thermal_expansion and m.poisson_ratio


# --------------------------------------------------------------------- #
#  The coupled solve
# --------------------------------------------------------------------- #
def test_joint_design_finds_feasible_large_low_pc_engine(prop, big_contour):
    """A large engine at low Pc with a GRCop-84 liner is jointly feasible:
    the optimizer returns a wall thickness inside its own band.

    Pc is low (1 MPa) because the structural screen now uses the FULL
    station-wise coolant-gas differential from the hydraulic march (SP-125
    eq. 4-31 inner-shell compressive stress over the whole contour, not a
    0.5·Pc throat guess), which is genuinely tighter — a copper liner only
    clears yield FoS=1 at modest chamber pressure."""
    r = joint_wall_channel_design(
        big_contour, prop, 1.0e6, material="grcop-84", mixture_ratio=2.0,
        thermal_margin=1.05, structural_fos=1.0, dp_budget_bar=300.0,
        channel_height=0.005, t_hot_max=0.0025, n_t=5, n_w=3, n_count=4, n_iter=10)
    assert r["feasible"] is True
    assert r["t_hot"] is not None
    assert r["thermal_margin"] >= 1.05
    assert r["structural_margin"] >= 1.0
    assert r["pressure_drop_bar"] <= 300.0
    # Coolant came from the cycle, not an input.
    assert r["mdot_cool"] == pytest.approx(r["mdot_total"] / 3.0, rel=1e-6)
    # The chosen thickness lies within the reported feasibility band.
    b = r["band"]
    assert b["feasible"] is True
    assert b["feasible_lo"] - 1e-12 <= r["t_hot"] <= b["feasible_hi"] + 1e-12


def test_thicker_wall_runs_hotter(prop, big_contour):
    """At fixed channels the thermal margin is non-increasing in t_hot —
    the conduction term q·t_w/k_w the optimizer trades against structure."""
    r = joint_wall_channel_design(
        big_contour, prop, 2.0e6, material="grcop-84", mixture_ratio=2.0,
        thermal_margin=1.05, structural_fos=1.0, channel_height=0.005,
        t_hot_max=0.0025, n_t=5, n_w=2, n_count=3, n_iter=10)
    slices = defaultdict(list)
    for c in r["candidates"]:
        slices[(c["N"], c["w"])].append(c)
    sl = sorted(next(iter(slices.values())), key=lambda c: c["t_hot"])
    tm = [c["thermal_margin"] for c in sl]
    assert len(tm) >= 3
    assert all(tm[i] >= tm[i + 1] - 1e-9 for i in range(len(tm) - 1))


def test_high_pc_copper_throat_is_stress_limited(prop, big_contour):
    """At high Pc the eq. 4-31 thermal + pressure stress exceeds copper's
    yield — the optimizer reports infeasible and names a stress/thermal
    binder.  GRCop fatigue is also evaluated from the sourced NASA direct
    total-strain/life regression."""
    r = joint_wall_channel_design(
        big_contour, prop, 12.0e6, material="grcop-84", mixture_ratio=2.6,
        thermal_margin=1.1, structural_fos=1.2, dp_budget_bar=400.0,
        channel_height=0.004, n_t=4, n_w=2, n_count=3, n_iter=10)
    assert r["feasible"] is False
    assert "structural" in r["diagnosis"] or "thermal" in r["diagnosis"]
    # Best-effort design is still returned.
    assert r["t_hot"] is not None and r["channel_count"] is not None
    assert r["fatigue_cycles"] is not None
    assert r["fatigue_status"] == "sourced_screening_gate"


def test_objective_must_be_known(prop, big_contour):
    with pytest.raises(ValueError):
        joint_wall_channel_design(
            big_contour, prop, 2.0e6, material="grcop-84",
            objective="bogus", n_t=2, n_w=2, n_count=2, n_iter=4)
