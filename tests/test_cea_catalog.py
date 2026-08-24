"""Installed, hash-pinned CEA catalog resources and explicit sentinel wiring."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
from importlib import resources
from pathlib import Path

import numpy as np
import pytest

from raosim.mdo.engine import chamber_surfaces_for
from raosim.mdo.job import (
    ArchitectureSelection,
    EngineJobSpec,
    MissionRequest,
    ModelSelection,
    resolve_engine_job,
)
from raosim.mdo.properties import (
    CEA_CATALOG_MANIFEST_RESOURCE,
    CEA_CATALOG_SENTINEL,
    load_cea_catalog_manifest,
    load_chamber_surfaces,
    resolve_catalog_cea_table,
)
from raosim.mdo.schema import MissionSpec


_EXPECTED = {
    "LOX/RP-1": {
        "file": "lox_rp1_frozen_v1.npz",
        "hash": "a65fd05ca3a9a5254879037a96bddb5bd6854e7c5bef434b9a87fffa780e8fdd",
        "domain": (1.5e6, 6.0e6, 1.6, 3.4),
        "grid": (17, 21),
    },
    "LOX/LCH4": {
        "file": "lox_lch4_frozen_v1.npz",
        "hash": "2d4ea43783bc99ac1eed43aacdedb82d33f9dcf4167e5c933c0ccb62b065192e",
        "domain": (1.5e6, 6.0e6, 2.2, 4.4),
        "grid": (17, 21),
    },
    "LOX/LH2": {
        "file": "lox_lh2_frozen_v1.npz",
        "hash": "33ffca0b8693dc651efaa696fea296c3c1a5fd03d8428ecfcc0ef1db6fce709c",
        "domain": (1.5e6, 6.0e6, 3.5, 8.0),
        "grid": (17, 25),
    },
    "N2O4/MMH": {
        "file": "n2o4_mmh_frozen_v1.npz",
        "hash": "e039f44befc96484f74b76ad0696fea8e4e849517c79e3325c6c237612ffa9b7",
        "domain": (1.5e6, 6.0e6, 1.3, 2.8),
        "grid": (17, 21),
    },
}


def _resource(relative_path: str):
    item = resources.files("raosim")
    for part in relative_path.split("/"):
        item = item.joinpath(part)
    return item


def test_installed_manifest_and_all_declared_table_resources_are_present():
    """Uses importlib.resources so the same assertion runs after wheel install."""

    manifest_resource = _resource(CEA_CATALOG_MANIFEST_RESOURCE)
    assert manifest_resource.is_file()
    manifest = load_cea_catalog_manifest()
    assert manifest["manifest_schema_version"] == 1
    assert manifest["catalog_version"] == "2026-08-22"
    assert {entry["propellant"] for entry in manifest["tables"]} == set(
        _EXPECTED
    )
    for expected in _EXPECTED.values():
        assert _resource(f"data/cea_tables/{expected['file']}").is_file()


@pytest.mark.parametrize("propellant", tuple(_EXPECTED))
def test_catalog_resolver_pins_identity_domain_grid_and_validation(propellant):
    expected = _EXPECTED[propellant]
    resolved = resolve_catalog_cea_table(propellant)
    surfaces = load_chamber_surfaces(
        CEA_CATALOG_SENTINEL,
        expected_propellant=propellant,
        require_of_dependence=True,
    )

    assert resolved.resource_name == expected["file"]
    assert resolved.table_content_sha256 == expected["hash"]
    assert surfaces.content_sha256 == expected["hash"]
    assert surfaces.provenance == (
        f"cea_frozen_catalog:{propellant}:{expected['file']}"
    )
    pc_grid = np.asarray(surfaces.gamma.xg)
    of_grid = np.asarray(surfaces.gamma.yg)
    assert (pc_grid.size, of_grid.size) == expected["grid"]
    assert (pc_grid[0], pc_grid[-1], of_grid[0], of_grid[-1]) == pytest.approx(
        expected["domain"]
    )
    validation = resolved.manifest_entry["validation"]
    assert validation["status"] == "passed"
    assert validation["complete_cell_coverage"] is True
    assert validation["cells_checked"] == (
        (expected["grid"][0] - 1) * (expected["grid"][1] - 1)
    )


def test_path_sentinel_resolves_but_empty_mission_keeps_constant_default():
    catalog = load_chamber_surfaces(
        Path(CEA_CATALOG_SENTINEL), expected_propellant="LOX/RP-1"
    )
    assert catalog.content_sha256 == _EXPECTED["LOX/RP-1"]["hash"]

    default_mission = MissionSpec.for_propellant("LOX/RP-1", 13.0e3)
    assert default_mission.cea_table_path == ""
    assert "constant_fallback" in chamber_surfaces_for(default_mission).provenance


def test_catalog_resolution_requires_context_and_refuses_uncertified_pairs():
    with pytest.raises(ValueError, match="requires a selected mission propellant"):
        load_chamber_surfaces(CEA_CATALOG_SENTINEL)
    with pytest.raises(ValueError, match="298.15 K gas-enthalpy.*184.4 K liquid"):
        load_chamber_surfaces(
            CEA_CATALOG_SENTINEL, expected_propellant="N2O/Ethanol"
        )
    with pytest.raises(ValueError, match="no validated.*Foo/Bar"):
        resolve_catalog_cea_table("Foo/Bar")


@pytest.mark.parametrize(
    "alias,canonical",
    (
        ("kerolox", "LOX/RP-1"),
        ("methalox", "LOX/LCH4"),
        ("hydrolox", "LOX/LH2"),
        ("mmh", "N2O4/MMH"),
    ),
)
def test_public_catalog_aliases_resolve_consistently(alias, canonical):
    resolved = resolve_catalog_cea_table(alias)
    surfaces = load_chamber_surfaces(
        CEA_CATALOG_SENTINEL,
        expected_propellant=alias,
        require_of_dependence=True,
    )
    assert resolved.propellant == canonical
    assert resolved.table_content_sha256 == _EXPECTED[canonical]["hash"]
    assert surfaces.content_sha256 == _EXPECTED[canonical]["hash"]
    assert surfaces.provenance.startswith(f"cea_frozen_catalog:{canonical}:")


@pytest.mark.parametrize(
    "custom_name,base_name",
    (
        ("O2/Kerosene", "LOX/RP-1"),
        # This exact key is also a public alias.  Runtime registration must not
        # be erased by alias expansion and mistaken for the built-in methalox.
        ("LOX/CH4", "LOX/LCH4"),
    ),
)
def test_user_registered_alias_equivalent_pair_cannot_inherit_certification(
    custom_name, base_name
):
    from raosim.mdo.propellants import (
        get_propellant,
        register_propellant,
        unregister_propellant,
    )

    evidence = {
        key: "catalog-boundary regression fixture"
        for key in (
            "chamber_gas",
            "l_star_basis",
            "efficiency_basis",
            "coolant_properties",
            "wall_limit_basis",
            "material_compatibility",
        )
    }
    stored = register_propellant(
        replace(get_propellant(base_name), name=custom_name),
        evidence=evidence,
    )
    try:
        assert stored.provenance == "user_registered"
        with pytest.raises(
            ValueError, match="refuses user-registered.*cannot be inherited"
        ):
            resolve_catalog_cea_table(custom_name)
        with pytest.raises(ValueError, match="refuses user-registered"):
            load_chamber_surfaces(
                CEA_CATALOG_SENTINEL,
                expected_propellant=custom_name,
            )
    finally:
        unregister_propellant(custom_name)


def test_catalog_resolver_rejects_manifest_identity_drift(monkeypatch):
    import raosim.mdo.properties as properties

    tampered = deepcopy(load_cea_catalog_manifest())
    tampered["tables"][0]["table"]["content_sha256"] = "0" * 64
    monkeypatch.setattr(properties, "load_cea_catalog_manifest", lambda: tampered)
    with pytest.raises(ValueError, match="content SHA-256 mismatch"):
        resolve_catalog_cea_table("LOX/RP-1")


@pytest.mark.parametrize(
    "ratio",
    (float("nan"), float("inf"), float("-inf"), -1.0e-12, 1.0000001, True),
)
def test_catalog_resolver_rejects_invalid_worst_tolerance_ratio(
    monkeypatch, ratio
):
    import raosim.mdo.properties as properties

    tampered = deepcopy(load_cea_catalog_manifest())
    tampered["tables"][0]["validation"]["worst_tolerance_ratio"] = ratio
    monkeypatch.setattr(properties, "load_cea_catalog_manifest", lambda: tampered)
    with pytest.raises(ValueError, match=r"finite and within \[0, 1\]"):
        resolve_catalog_cea_table("LOX/RP-1")


@pytest.mark.parametrize("ratio", (0.0, 1.0))
def test_catalog_resolver_accepts_inclusive_tolerance_ratio_bounds(
    monkeypatch, ratio
):
    import raosim.mdo.properties as properties

    changed = deepcopy(load_cea_catalog_manifest())
    changed["tables"][0]["validation"]["worst_tolerance_ratio"] = ratio
    monkeypatch.setattr(properties, "load_cea_catalog_manifest", lambda: changed)
    assert resolve_catalog_cea_table("kerolox").propellant == "LOX/RP-1"


def _mdo_args(**updates):
    values = dict(
        target_thrust=13.0e3,
        pump_rpm=None,
        engine_mdo_ambient=None,
        burn_time=None,
        design_margins=False,
        mdo_chamber_property_table=None,
        mdo_pc_search_min_pa=None,
        mdo_pc_search_max_pa=None,
        mdo_propellant=None,
        material=None,
        jacket_material=None,
        mdo_of_mode=None,
        mixture_ratio=None,
    )
    values.update(updates)
    return argparse.Namespace(**values)


def test_cli_path_sentinel_stays_portable_and_selects_mission_pair():
    from raosim.requirements import MixtureRatioMode
    from raosim.run_nozzle import _mission_from_mdo_args

    mission, mode = _mission_from_mdo_args(
        _mdo_args(
            mdo_chamber_property_table=Path(CEA_CATALOG_SENTINEL),
            mdo_propellant="LOX/LCH4",
            mdo_of_mode="optimize",
        ),
        optimization_capable=True,
    )
    assert mode is MixtureRatioMode.OPTIMIZE
    assert mission.cea_table_path == CEA_CATALOG_SENTINEL
    assert mission.propellant_name == "LOX/LCH4"
    of_spec = mission.of_design_space()[0]
    assert (of_spec.lower, of_spec.upper) == pytest.approx((2.2, 4.4))


def test_requirement_api_catalog_sentinel_resolves_selected_propellant():
    from raosim.requirements import (
        EngineRequirement,
        MixtureRatioMode,
        RequirementAnalysisConfig,
        resolve_requirement,
    )

    resolved = resolve_requirement(EngineRequirement(
        thrust=13.0e3,
        thrust_condition="sea_level",
        flight_duration=10.0,
        propellant="LOX/LH2",
        mixture_ratio_mode=MixtureRatioMode.OPTIMIZE,
        analysis_config=RequirementAnalysisConfig(
            chamber_property_table=CEA_CATALOG_SENTINEL
        ),
    ))
    assert resolved.mission.cea_table_path == CEA_CATALOG_SENTINEL
    assert resolved.mission.propellant_name == "LOX/LH2"
    of_spec = resolved.mission.of_design_space()[0]
    assert (of_spec.lower, of_spec.upper) == pytest.approx((3.5, 8.0))


def test_job_resolution_eagerly_validates_catalog_selection():
    valid = EngineJobSpec(
        mission=MissionRequest(thrust="13 kN"),
        architecture=ArchitectureSelection(propellant="N2O4/MMH"),
        models=ModelSelection(chamber_property_table=CEA_CATALOG_SENTINEL),
    )
    resolved = resolve_engine_job(valid)
    assert resolved.mission.cea_table_path == CEA_CATALOG_SENTINEL
    assert resolved.mission.propellant_name == "N2O4/MMH"

    blocked = EngineJobSpec(
        mission=MissionRequest(thrust="13 kN"),
        architecture=ArchitectureSelection(propellant="N2O/Ethanol"),
        models=ModelSelection(chamber_property_table=CEA_CATALOG_SENTINEL),
    )
    with pytest.raises(ValueError, match="select and document the oxidizer"):
        resolve_engine_job(blocked)
