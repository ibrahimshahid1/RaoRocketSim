"""Materials catalog + SP-125 eq. 4-31 wall-stress relation.

The catalog (``raosim.materials``) feeds one coherent set of wall
properties into channel sizing (k -> wall temperature) and the structural
screen (E, a, v -> thermal stress), grounded in NASA SP-125
(``propulsion_texts/19710019929.pdf``).
"""
from __future__ import annotations

import math

import pytest

from raosim.materials import (
    CATEGORY_COPPER_LINER, get_material, list_materials, material_names,
    material_table,
)


def test_every_material_has_physical_properties():
    for m in list_materials():
        assert m.conductivity > 0
        assert m.yield_strength > 0
        assert m.max_temperature > 0
        assert m.elastic_modulus > 0
        assert m.thermal_expansion > 0
        assert 0.0 < m.poisson_ratio < 0.5
        assert m.density > 0
        assert m.max_heat_flux > 0
        # the service limit must sit below the melting point
        assert m.max_temperature < m.melting_point
        assert m.source  # provenance recorded


def test_names_and_aliases_are_unique():
    names = material_names()
    assert len(names) == len(set(names))


def test_copper_liners_outconduct_the_superalloys():
    """The whole design trade: copper liners conduct ~20x better than the
    Ni-base superalloys (so they cool the wall) but tolerate far less
    temperature.  Catalog must encode both halves."""
    liners = [m for m in list_materials() if m.is_liner]
    superalloys = [m for m in list_materials() if m.category == "superalloy"]
    assert liners and superalloys
    assert min(m.conductivity for m in liners) > 10 * max(
        m.conductivity for m in superalloys)
    # ...and the superalloys take more heat than the copper liners
    assert max(m.max_temperature for m in superalloys) > max(
        m.max_temperature for m in liners)


@pytest.mark.parametrize("query,expected", [
    ("grcop-84", "GRCop-84"), ("GRCop84", "GRCop-84"), ("grcop_84", "GRCop-84"),
    ("narloy", "NARloy-Z"), ("narloy-z", "NARloy-Z"),
    ("inconel718", "Inconel 718"), ("in718", "Inconel 718"), ("718", "Inconel 718"),
    ("inconel-x", "Inconel X-750"), ("x750", "Inconel X-750"),
    ("316l", "Stainless 316L"), ("ss316", "Stainless 316L"),
    ("cu", "OFHC Copper"), ("copper", "OFHC Copper"),
    ("ni", "Nickel 200"), ("cucrzr", "CuCrZr"),
])
def test_get_material_resolves_aliases_case_insensitively(query, expected):
    assert get_material(query).name == expected
    assert get_material(query.upper()).name == expected


def test_get_material_unknown_raises_with_available_names():
    with pytest.raises(KeyError) as exc:
        get_material("unobtainium")
    assert "GRCop-84" in str(exc.value)  # the message lists the catalog


def test_grcop84_is_the_high_temperature_copper():
    """GRCop-84 is the modern AM liner: higher service T than NARloy-Z."""
    assert get_material("grcop-84").max_temperature > get_material(
        "narloy-z").max_temperature


def test_material_spec_from_catalog_carries_structural_fields():
    from raosim.design import MaterialSpec
    ms = MaterialSpec.from_catalog("narloy-z")
    m = get_material("narloy-z")
    assert ms.name == "NARloy-Z"
    assert ms.conductivity == m.conductivity
    assert ms.yield_strength == m.yield_strength
    assert ms.max_temperature == m.max_temperature
    # the extra fields the eq. 4-31 stress check needs
    assert ms.elastic_modulus == m.elastic_modulus
    assert ms.thermal_expansion == m.thermal_expansion
    assert ms.poisson_ratio == m.poisson_ratio
    assert ms.density == m.density


def test_material_table_lists_every_material():
    table = material_table()
    for name in material_names():
        assert name in table
    assert "W/m.K" in table  # units header present


# ---------------------------------------------------------------------
#  SP-125 eq. 4-31 combined wall stress (printed p. 109)
#  S_c = (p_co - p_g) R / t  +  E a q t / (2 (1 - v) k)
# ---------------------------------------------------------------------

def _stress(**over):
    from raosim.physics import coaxial_shell_wall_stress
    kw = dict(pressure_differential=5e6, inner_radius=0.05, wall_thickness=0.001,
              heat_flux=50e6, elastic_modulus=140e9, thermal_expansion=16.5e-6,
              poisson_ratio=0.33, conductivity=285.0, yield_strength=186e6)
    kw.update(over)
    return coaxial_shell_wall_stress(**kw)


def test_eq431_matches_the_closed_form():
    s = _stress()
    p = 5e6 * 0.05 / 0.001
    th = 140e9 * 16.5e-6 * 50e6 * 0.001 / (2 * (1 - 0.33) * 285.0)
    assert s["pressure_stress"] == pytest.approx(p, rel=1e-12)
    assert s["thermal_stress"] == pytest.approx(th, rel=1e-12)
    assert s["combined_stress"] == pytest.approx(abs(p) + th, rel=1e-12)
    assert s["stress_margin"] == pytest.approx(186e6 / (abs(p) + th), rel=1e-12)


def test_thermal_stress_grows_with_flux_and_thickness():
    """The thermal term is E a q t / (2(1-v)k): linear in heat flux q and
    in wall thickness t (a thicker wall runs a steeper gradient).  This is
    the squeeze that opposes thinning the wall for conduction."""
    base = _stress(heat_flux=40e6, wall_thickness=0.001)
    hotter = _stress(heat_flux=80e6, wall_thickness=0.001)
    thicker = _stress(heat_flux=40e6, wall_thickness=0.002)
    assert hotter["thermal_stress"] == pytest.approx(2 * base["thermal_stress"], rel=1e-9)
    assert thicker["thermal_stress"] == pytest.approx(2 * base["thermal_stress"], rel=1e-9)


def test_pressure_stress_falls_with_thickness():
    thin = _stress(wall_thickness=0.001)
    thick = _stress(wall_thickness=0.002)
    assert thick["pressure_stress"] == pytest.approx(0.5 * thin["pressure_stress"], rel=1e-9)


def test_high_conductivity_wall_has_lower_thermal_stress():
    """Copper's high k relieves thermal stress (q t / k); a low-k
    superalloy at the same flux/thickness carries far more."""
    copper = _stress(conductivity=320.0, elastic_modulus=108e9,
                     thermal_expansion=18.0e-6, poisson_ratio=0.34)
    inconel = _stress(conductivity=15.0, elastic_modulus=200e9,
                      thermal_expansion=13.0e-6, poisson_ratio=0.29)
    assert inconel["thermal_stress"] > 5 * copper["thermal_stress"]
