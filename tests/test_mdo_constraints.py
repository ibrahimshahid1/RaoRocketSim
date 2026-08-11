"""Single-manifest and tri-state feasibility contract gates."""

from __future__ import annotations

import numpy as np

from raosim.mdo.constraints import (
    CONSTRAINT_MANIFEST,
    ENGINE_CONSTRAINT_SPECS,
    OPTIMIZER_CONSTRAINT_NAMES,
    constraint_metadata,
    status_from_rows,
)
from raosim.mdo.nlp import DEFAULT_ENFORCED
from raosim.mdo.properties import constant_chamber_surfaces
from raosim.mdo.schema import MissionSpec


def _constant_surfaces(mission):
    return constant_chamber_surfaces(
        gamma=mission.gamma, Tc=mission.Tc, R_gas=mission.R_gas
    )


def test_manifest_names_and_engine_keys_are_unique():
    names = [row.name for row in CONSTRAINT_MANIFEST]
    keys = [row.engine_key for row in ENGINE_CONSTRAINT_SPECS]
    assert len(names) == len(set(names))
    assert len(keys) == len(set(keys))


def test_nlp_default_rows_are_generated_from_manifest():
    assert DEFAULT_ENFORCED == OPTIMIZER_CONSTRAINT_NAMES
    roles = {row.name: row.optimizer_role for row in CONSTRAINT_MANIFEST}
    assert roles["engine_residual"] == "final_gate"
    assert roles["cooling_residual"] == "final_gate"
    assert "engine_residual" not in DEFAULT_ENFORCED
    assert "cooling_residual" not in DEFAULT_ENFORCED


def test_residuals_remain_mandatory_final_gates():
    required = {
        row.name for row in CONSTRAINT_MANIFEST
        if row.optimizer_role == "final_gate"
    }
    assert {
        "engine_residual", "cooling_residual", "solver_status", "finite",
        "coolant_htd",
    } <= required


def test_rp1_uses_coking_and_marks_htd_not_applicable():
    mission = MissionSpec.for_propellant("LOX/RP-1", 13.0e3)
    app, available, required, _ = constraint_metadata(
        mission, _constant_surfaces(mission)
    )
    index = {row.name: i for i, row in enumerate(ENGINE_CONSTRAINT_SPECS)}
    assert app[index["coking"]] and available[index["coking"]]
    assert not app[index["coolant_htd"]]
    assert not required[index["coolant_htd"]]
    assert not app[index["property_domain"]]


def test_methane_requires_unavailable_htd_and_does_not_use_coking():
    mission = MissionSpec.for_propellant("LOX/LCH4", 13.0e3)
    app, available, required, _ = constraint_metadata(
        mission, _constant_surfaces(mission)
    )
    index = {row.name: i for i, row in enumerate(ENGINE_CONSTRAINT_SPECS)}
    assert not app[index["coking"]]
    assert app[index["coolant_htd"]]
    assert not available[index["coolant_htd"]]
    assert required[index["coolant_htd"]]


def test_tri_state_reduction_preserves_fail_unknown_and_pass():
    values = np.array([1.0, 1.0])
    applicable = np.array([True, True])
    required = np.array([True, True])
    assert status_from_rows(
        values, applicable, np.array([True, True]), required
    ) == "pass"
    assert status_from_rows(
        values, applicable, np.array([True, False]), required
    ) == "unknown"
    # A known violation takes precedence over a separate unavailable row.
    assert status_from_rows(
        np.array([-1.0, 1.0]), applicable,
        np.array([True, False]), required,
    ) == "fail"


def test_uncatalogued_propellant_cannot_silently_drop_the_wall_side_gate():
    """An unresolved coolant identity must reduce to unknown, never to pass.

    Regression: the metadata builder used to fall back to the raw propellant
    string, so a mistyped methane/hydrogen mission reported ``coking`` as an
    applicable, available, satisfied row and lost its governing HTD gate.
    """
    from dataclasses import replace

    from raosim.mdo.constraints import ConstraintReasonCode, reason_text

    mission = replace(
        MissionSpec.for_propellant("LOX/LCH4", 13.0e3),
        propellant_name="LOX/LCH4-typo",
    )
    app, available, required, reasons = constraint_metadata(
        mission, _constant_surfaces(mission)
    )
    index = {row.name: i for i, row in enumerate(ENGINE_CONSTRAINT_SPECS)}

    for name in ("coking", "coolant_htd"):
        i = index[name]
        assert app[i], f"{name} must stay applicable while the coolant is unknown"
        assert not available[i], f"{name} cannot be available without a coolant"
        assert required[i]
        assert reasons[i] == int(ConstraintReasonCode.COOLANT_UNRESOLVED)
        assert "not in the catalog" in reason_text(
            int(reasons[i]), ENGINE_CONSTRAINT_SPECS[i], mission
        )

    # Every margin passing must still not produce a feasible verdict.
    values = np.ones(len(ENGINE_CONSTRAINT_SPECS))
    assert status_from_rows(values, app, available, required) == "unknown"


def test_catalogued_propellants_keep_their_resolved_wall_side_mechanism():
    """The unresolved-coolant guard must not perturb known propellants."""
    expected = {
        "LOX/RP-1": ("coking", "coolant_htd"),
        "N2O4/MMH": ("coking", "coolant_htd"),
        "LOX/LCH4": ("coolant_htd", "coking"),
        "LOX/LH2": ("coolant_htd", "coking"),
    }
    for propellant, (governing, inactive) in expected.items():
        mission = MissionSpec.for_propellant(propellant, 13.0e3)
        app, _, _, _ = constraint_metadata(mission, _constant_surfaces(mission))
        index = {row.name: i for i, row in enumerate(ENGINE_CONSTRAINT_SPECS)}
        assert app[index[governing]], f"{propellant}: {governing} must apply"
        assert not app[index[inactive]], f"{propellant}: {inactive} must not apply"


def test_new_policy_sensitive_rows_have_exact_source_ids():
    source = {row.name: row.source_id for row in CONSTRAINT_MANIFEST}
    assert "PDF137-printed128" in (source["chug"] or "")
    assert "PDF293-294" in (source["chug"] or "")
    assert source["coolant_htd"] == (
        "Nasuti-Pizzarelli-2021-sec-2.2-Eq9-PDF6"
    )
