"""The single typed catalog->traced-property mapper, and its atomicity.

Regression context: a host-side material label reached the traditional
pipeline through ``MaterialSpec.from_catalog`` while the differentiable core
kept flat class defaults, so ``--material GRCop-84`` optimized a
NARloy-Z-class liner.  The defaults were not even one alloy -- NARloy-Z
conductivity and density against a CuCrZr-class allowable.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import pytest

from raosim.materials import get_material
from raosim.mdo.material_map import (
    CLOSEOUT_MISSION_FIELDS,
    LINER_MISSION_FIELDS,
    MaterialCoverageError,
    closeout_mission_fields,
    liner_mission_fields,
    resolve_material_selection,
)
from raosim.mdo.schema import MissionSpec


def test_liner_map_sets_every_owned_field_from_one_catalog_record():
    mat = get_material("GRCop-84")
    fields = liner_mission_fields(mat, structural_fos=1.5)
    assert set(LINER_MISSION_FIELDS) <= set(fields)
    assert fields["k_wall"] == pytest.approx(mat.conductivity)
    assert fields["rho_wall"] == pytest.approx(mat.density)
    assert fields["liner_E"] == pytest.approx(mat.elastic_modulus)
    assert fields["liner_alpha"] == pytest.approx(mat.thermal_expansion)
    assert fields["liner_poisson"] == pytest.approx(mat.poisson_ratio)
    assert fields["liner_T_wg_max"] == pytest.approx(mat.max_temperature)


def test_factor_of_safety_is_applied_exactly_once():
    """``liner_sigma_allow`` is post-FOS; the factor is retained separately."""
    mat = get_material("GRCop-84")
    fields = liner_mission_fields(mat, structural_fos=1.5)
    assert fields["liner_sigma_allow"] == pytest.approx(mat.yield_strength / 1.5)
    assert fields["liner_structural_fos"] == pytest.approx(1.5)
    # Recovering yield from the pair must return the catalog value.
    recovered = fields["liner_sigma_allow"] * fields["liner_structural_fos"]
    assert recovered == pytest.approx(mat.yield_strength)


def test_closeout_yield_is_raw_because_the_jacket_applies_its_own_factor():
    mat = get_material("Inconel 718")
    fields = closeout_mission_fields(mat)
    assert set(CLOSEOUT_MISSION_FIELDS) == set(fields)
    assert fields["closeout_sigma_yield"] == pytest.approx(mat.yield_strength)


def test_inconel718_closeout_reproduces_the_documented_defaults():
    """The mapper must agree with the values the class defaults claimed."""
    fields = closeout_mission_fields(get_material("Inconel 718"))
    assert fields["rho_closeout"] == pytest.approx(8190.0)
    assert fields["closeout_sigma_yield"] == pytest.approx(1035.0e6)
    assert fields["closeout_E"] == pytest.approx(200.0e9)
    assert fields["closeout_poisson"] == pytest.approx(0.29)


def test_incomplete_material_is_rejected_rather_than_partially_applied():
    """A half-applied alloy exists in no catalog; the call must raise."""
    broken = replace(get_material("GRCop-84"), elastic_modulus=None)
    with pytest.raises(MaterialCoverageError, match="elastic_modulus"):
        liner_mission_fields(broken)

    nonphysical = replace(get_material("GRCop-84"), conductivity=-1.0)
    with pytest.raises(MaterialCoverageError, match="non-physical"):
        liner_mission_fields(nonphysical)


def test_unknown_material_is_rejected():
    with pytest.raises(KeyError):
        resolve_material_selection(liner="Unobtainium", closeout="Inconel 718")


def test_for_material_retargets_every_traced_property():
    mission = MissionSpec.for_material("GRCop-84", 13.0e3)
    mat = get_material("GRCop-84")
    assert mission.liner_material_name == "GRCop-84"
    assert mission.closeout_material_name == "Inconel 718"
    assert mission.k_wall == pytest.approx(mat.conductivity)
    assert mission.rho_wall == pytest.approx(mat.density)
    assert mission.liner_E == pytest.approx(mat.elastic_modulus)
    assert mission.liner_T_wg_max == pytest.approx(mat.max_temperature)


def test_default_mission_does_not_claim_a_material():
    """Unattributed class defaults must not imply a catalog selection."""
    mission = MissionSpec.for_thrust(13.0e3)
    assert mission.liner_material_name is None
    assert mission.closeout_material_name is None


def test_liner_and_closeout_are_never_inherited_from_one_another():
    """SP-8087 sec. 2.1.3.1: a soft liner inside a hardenable jacket."""
    mission = MissionSpec.for_material("GRCop-84", 13.0e3)
    assert mission.closeout_material_name != mission.liner_material_name
    assert mission.closeout_sigma_yield > mission.liner_sigma_allow

    with pytest.raises(ValueError, match="explicit"):
        MissionSpec.for_thrust(13.0e3).with_materials(liner="GRCop-84")


def test_for_material_composes_with_for_propellant():
    mission = MissionSpec.for_material(
        "GRCop-84", 13.0e3, propellant="LOX/LCH4"
    )
    assert mission.propellant_name == "LOX/LCH4"
    assert mission.liner_material_name == "GRCop-84"
    assert mission.k_wall == pytest.approx(get_material("GRCop-84").conductivity)


def test_distinct_materials_produce_distinct_traced_walls():
    """The whole point: a label change must move the traced physics."""
    walls = {}
    for name in ("GRCop-84", "NARloy-Z", "CuCrZr", "OFHC Copper"):
        m = MissionSpec.for_material(name, 13.0e3)
        walls[name] = (m.k_wall, m.rho_wall, m.liner_E, m.liner_T_wg_max)
    assert len(set(walls.values())) == len(walls)


def test_cli_material_flag_reaches_the_mdo_mission():
    """Regression: --material used to change only the traditional pipeline."""
    from raosim.run_nozzle import _mission_from_mdo_args

    def args(**kw):
        base = dict(
            target_thrust=13.0e3, pump_rpm=None, engine_mdo_ambient=None,
            burn_time=None, design_margins=False,
            mdo_chamber_property_table=None, mdo_pc_search_min_pa=None,
            mdo_pc_search_max_pa=None, mdo_propellant=None, material=None,
            jacket_material=None, mdo_of_mode=None, mixture_ratio=None,
            engine_mdo_optimize=False,
        )
        base.update(kw)
        return argparse.Namespace(**base)

    mission, _ = _mission_from_mdo_args(
        args(material="grcop-84"), optimization_capable=False
    )
    mat = get_material("GRCop-84")
    assert mission.liner_material_name == "GRCop-84"
    assert mission.k_wall == pytest.approx(mat.conductivity)
    assert mission.liner_T_wg_max == pytest.approx(mat.max_temperature)

    jacketed, _ = _mission_from_mdo_args(
        args(material="grcop-84", jacket_material="inconel x-750"),
        optimization_capable=False,
    )
    assert jacketed.closeout_material_name == "Inconel X-750"
    assert jacketed.closeout_sigma_yield == pytest.approx(
        get_material("Inconel X-750").yield_strength
    )

    with pytest.raises(KeyError):
        _mission_from_mdo_args(
            args(material="unobtainium"), optimization_capable=False
        )
