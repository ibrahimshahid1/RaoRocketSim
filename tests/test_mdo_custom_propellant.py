"""Runtime registration of a custom propellant, and the wall-side mechanism.

The defect this closes is not only "custom propellants are unreachable". It is
that the coolant-side mechanism was inferred from a **coolant name string**:
``"LNG"``, ``"ethane"`` and ``"subcooled methane"`` all miss the built-in
supercritical list and would have been handed the hydrocarbon coking limit,
silently, for fluids whose governing physics is deterioration.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from raosim.mdo.constraints import (
    ENGINE_CONSTRAINT_SPECS,
    ConstraintReasonCode,
    constraint_metadata,
)
from raosim.mdo.engine import chamber_surfaces_for
from raosim.mdo.propellants import (
    PropellantCoverageError,
    PropellantSpec,
    WallSideMechanism,
    get_propellant,
    register_propellant,
    registered_propellants,
    unregister_propellant,
)
from raosim.mdo.schema import MissionSpec

_EVIDENCE = {
    key: "unit-test source, not literature"
    for key in (
        "chamber_gas", "l_star_basis", "efficiency_basis",
        "coolant_properties", "wall_limit_basis", "material_compatibility",
    )
}


@pytest.fixture
def methane_like():
    return get_propellant("LOX/LCH4")


@pytest.fixture
def cleanup():
    created: list[str] = []
    yield created
    for name in created:
        unregister_propellant(name)


def _gates(propellant: str):
    mission = replace(MissionSpec(), propellant_name=propellant)
    app, avail, req, reasons = constraint_metadata(
        mission, chamber_surfaces_for(mission)
    )
    index = {spec.name: i for i, spec in enumerate(ENGINE_CONSTRAINT_SPECS)}
    return {
        name: (
            bool(app[index[name]]),
            bool(avail[index[name]]),
            ConstraintReasonCode(int(reasons[index[name]])),
        )
        for name in ("coking", "coolant_htd")
    }


def test_registered_propellant_becomes_solvable(methane_like, cleanup):
    """``solve_engine`` used to reject any uncatalogued propellant outright."""
    import jax.numpy as jnp

    from raosim.mdo.engine import solve_engine
    from raosim.mdo.schema import DesignVector, default_design_space

    register_propellant(
        replace(methane_like, name="LOX/LNG", coolant_name="LNG",
                wall_side_mechanism=WallSideMechanism.SUPERCRITICAL_HTD),
        evidence=_EVIDENCE,
    )
    cleanup.append("LOX/LNG")
    assert "lox/lng" in registered_propellants()

    mission = MissionSpec.for_propellant("LOX/LNG", 13.0e3).with_materials(
        liner="GRCop-84", closeout="Inconel 718"
    )
    layout = mission.design_layout()
    values = {v.name: v.ref() for v in default_design_space(mission)}
    x = DesignVector.from_active_array(
        jnp.asarray([values[n] for n in layout.active_names]),
        layout, fixed_of=mission.OF,
    )
    result = solve_engine(x, mission, surfaces=chamber_surfaces_for(mission))
    assert bool(result.solver_converged)
    assert float(result.Isp) > 0.0


def test_declared_mechanism_beats_the_coolant_name_list(methane_like, cleanup):
    """The safety-critical case: LNG is methane, but misses the name list."""
    from raosim.mdo.coolant_htd import htd_is_applicable

    # The latent defect, demonstrated directly.
    assert htd_is_applicable("methane") is True
    assert htd_is_applicable("LNG") is False
    assert htd_is_applicable("subcooled methane") is False

    register_propellant(
        replace(methane_like, name="LOX/LNG", coolant_name="LNG",
                wall_side_mechanism=WallSideMechanism.SUPERCRITICAL_HTD),
        evidence=_EVIDENCE,
    )
    cleanup.append("LOX/LNG")

    gates = _gates("LOX/LNG")
    # Coking must NOT govern, and HTD must be applicable-but-unavailable.
    assert gates["coking"][0] is False
    assert gates["coolant_htd"] == (
        True, False, ConstraintReasonCode.MODEL_UNAVAILABLE
    )


def test_unknown_mechanism_reduces_to_unknown_never_a_pass(methane_like,
                                                           cleanup):
    register_propellant(
        replace(methane_like, name="LOX/Novel", coolant_name="novel-fluid",
                wall_side_mechanism=WallSideMechanism.UNKNOWN),
        evidence=_EVIDENCE,
    )
    cleanup.append("LOX/Novel")

    gates = _gates("LOX/Novel")
    for name in ("coking", "coolant_htd"):
        applicable, available, reason = gates[name]
        assert applicable and not available
        assert reason is ConstraintReasonCode.COOLANT_UNRESOLVED


def test_catalog_entries_are_unchanged_by_the_declaration_change():
    assert _gates("LOX/RP-1")["coking"][:2] == (True, True)
    assert _gates("LOX/RP-1")["coolant_htd"][0] is False
    assert _gates("LOX/LCH4")["coking"][0] is False
    assert _gates("LOX/LCH4")["coolant_htd"][:2] == (True, False)
    for name in ("LOX/RP-1", "LOX/LCH4", "LOX/LH2", "N2O4/MMH", "N2O/Ethanol"):
        assert get_propellant(name).wall_side_mechanism is not None


def test_registration_requires_an_explicit_mechanism(methane_like):
    with pytest.raises(PropellantCoverageError, match="must be declared"):
        register_propellant(
            replace(methane_like, name="LOX/A", wall_side_mechanism=None),
            evidence=_EVIDENCE,
        )


def test_coking_mechanism_requires_the_limit_that_is_the_constraint(
    methane_like,
):
    with pytest.raises(PropellantCoverageError, match="coolant_wall_limit_K"):
        register_propellant(
            replace(methane_like, name="LOX/B",
                    wall_side_mechanism=WallSideMechanism.COKING,
                    coolant_wall_limit_K=None),
            evidence=_EVIDENCE,
        )


def test_unsourced_numbers_cannot_become_a_design_basis(methane_like):
    with pytest.raises(PropellantCoverageError, match="missing evidence"):
        register_propellant(
            replace(methane_like, name="LOX/C",
                    wall_side_mechanism=WallSideMechanism.COKING),
            evidence={"chamber_gas": "somewhere"},
        )


def test_incomplete_or_nonphysical_properties_are_rejected(methane_like):
    with pytest.raises(PropellantCoverageError, match="rho_cool"):
        register_propellant(
            replace(methane_like, name="LOX/D", rho_cool=-1.0,
                    wall_side_mechanism=WallSideMechanism.COKING),
            evidence=_EVIDENCE,
        )


def test_a_custom_entry_cannot_shadow_validated_catalog_data(methane_like):
    with pytest.raises(PropellantCoverageError, match="shadows"):
        register_propellant(
            replace(methane_like, name="LOX/RP-1",
                    wall_side_mechanism=WallSideMechanism.COKING),
            evidence=_EVIDENCE,
        )


def test_registration_is_marked_user_supplied_not_repository_validated(
    methane_like, cleanup,
):
    """A report must be able to tell the two apart."""
    stored = register_propellant(
        replace(methane_like, name="LOX/LNG", coolant_name="LNG",
                wall_side_mechanism=WallSideMechanism.SUPERCRITICAL_HTD),
        evidence=_EVIDENCE,
    )
    cleanup.append("LOX/LNG")
    assert stored.provenance == "user_registered"
    assert stored.estimated is True
    assert get_propellant("LOX/RP-1").provenance == "catalog"


def test_unknown_propellant_error_points_at_the_registration_path():
    with pytest.raises(KeyError, match="register_propellant"):
        get_propellant("LOX/Nothing")


def test_a_job_document_can_carry_its_own_propellant(tmp_path, cleanup):
    """A job must be self-contained and replayable."""
    import json
    from dataclasses import asdict

    from lrekit.job_cli import _build_spec
    from raosim.mdo.job import resolve_engine_job

    payload = asdict(get_propellant("LOX/LCH4"))
    payload.update(
        name="LOX/LNG", coolant_name="LNG",
        wall_side_mechanism="supercritical_htd",
    )
    payload["evidence"] = dict(_EVIDENCE)
    document = {
        "mission": {"thrust": "13 kN"},
        "workflow": "optimize",
        "objective": {"metric": "masses.dry_mass_partial", "direction": "min"},
        "custom_propellants": [payload],
        "architecture": {"propellant": "LOX/LNG",
                         "liner_material": "GRCop-84"},
    }
    cleanup.append("LOX/LNG")

    job = resolve_engine_job(_build_spec(document))
    assert job.mission.propellant_name == "LOX/LNG"
    assert get_propellant("LOX/LNG").provenance == "user_registered"
    # And the document survives a JSON round trip.
    assert json.loads(json.dumps(document, default=str))
    # The compiled/saved job, not just the original caller's document, retains
    # the custom definition and changes identity when that definition changes.
    replay_document = job.spec.to_document()
    assert replay_document["custom_propellants"][0]["evidence"] == _EVIDENCE
    unregister_propellant("LOX/LNG")
    replay = resolve_engine_job(_build_spec(json.loads(json.dumps(replay_document))))
    assert replay.job_digest == job.job_digest
    assert replay.mission.propellant_name == "LOX/LNG"
