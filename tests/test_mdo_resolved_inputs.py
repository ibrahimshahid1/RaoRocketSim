"""The single host-side input contract (remediation item 9).

These gates encode the invariant the contract exists for: one resolved
definition of every convention, consumed by both pipelines, with no field
reconstructible from a default and no quantity re-derived in a second place.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import pytest

from raosim.mdo.engine import chamber_surfaces_for, solve_engine
from raosim.mdo.resolved_inputs import (
    RESOLVED_INPUTS_SCHEMA_VERSION,
    ResolvedEngineInputs,
    resolve_engine_inputs,
)
from raosim.mdo.schema import DesignVector, MissionSpec, default_design_space


def _solved(mission):
    layout = mission.design_layout()
    vals = {v.name: v.ref() for v in default_design_space(mission)}
    x = DesignVector.from_active_array(
        jnp.asarray([vals[n] for n in layout.active_names]),
        layout,
        fixed_of=mission.OF,
    )
    surfaces = chamber_surfaces_for(mission)
    return x, surfaces, solve_engine(x, mission, surfaces=surfaces)


def _resolve(mission, **kw):
    x, surfaces, r = _solved(mission)
    kw.setdefault("effective_of", float(r.OF))
    kw.setdefault("of_source", "mission_nominal")
    kw.setdefault("total_mass_flow", float(r.mdot))
    kw.setdefault("surfaces", surfaces)
    return resolve_engine_inputs(
        x.as_contract_dict(effective_of=r.OF), mission, **kw
    )


def test_contract_is_frozen_and_versioned():
    inputs = _resolve(MissionSpec.for_material("GRCop-84", 13.0e3))
    assert inputs.schema_version == RESOLVED_INPUTS_SCHEMA_VERSION
    with pytest.raises(Exception):
        inputs.target_thrust = 1.0  # type: ignore[misc]


def test_mass_flow_split_is_defined_once_and_uses_the_effective_of():
    """Re-deriving the split in a second place is how the optimum was lost."""
    mission = MissionSpec.for_material("GRCop-84", 13.0e3)
    inputs = _resolve(mission, effective_of=2.65, of_source="optimized")
    assert inputs.propellant.mixture_ratio == pytest.approx(2.65)
    assert inputs.propellant.mixture_ratio_source == "optimized"
    assert inputs.fuel_mass_flow == pytest.approx(
        inputs.total_mass_flow / 3.65
    )
    assert inputs.fuel_mass_flow + inputs.oxidizer_mass_flow == pytest.approx(
        inputs.total_mass_flow
    )
    # The mission nominal must not leak back in.
    assert inputs.propellant.mixture_ratio != pytest.approx(float(mission.OF))


def test_regen_and_film_branches_close_to_total_fuel_flow():
    inputs = _resolve(MissionSpec.for_material("GRCop-84", 13.0e3))
    inputs.assert_flow_closure()

    broken = replace(
        inputs,
        thermal=replace(inputs.thermal, film_mass_flow=inputs.thermal.film_mass_flow + 1.0),
    )
    with pytest.raises(ValueError, match="must equal total fuel"):
        broken.assert_flow_closure()


def test_factor_of_safety_is_recoverable_without_double_application():
    from raosim.materials import get_material

    inputs = _resolve(MissionSpec.for_material("GRCop-84", 13.0e3))
    assert inputs.material.liner_yield_strength == pytest.approx(
        get_material("GRCop-84").yield_strength
    )


def test_material_selection_is_carried_and_unresolved_defaults_are_flagged():
    resolved = _resolve(MissionSpec.for_material("GRCop-84", 13.0e3))
    assert resolved.material.liner_name == "GRCop-84"
    assert resolved.material.liner_selection_resolved
    assert "liner_material" not in resolved.unavailable

    unattributed = _resolve(MissionSpec.for_thrust(13.0e3))
    assert unattributed.material.liner_name is None
    assert not unattributed.material.liner_selection_resolved
    assert "liner_material" in unattributed.unavailable


def test_uncatalogued_coolant_identity_is_inferred_but_labelled_as_such():
    """Fuel-as-coolant is the architecture, not a guess at an unknown fluid.

    The inference must still be recorded, so nothing downstream can mistake it
    for a catalog-backed identity. Which wall-side mechanism governs stays
    ``unknown`` separately, through the constraint manifest.

    ``solve_engine`` rejects an uncatalogued propellant outright, so the
    resolver is exercised directly here: it is the layer that would have to
    stay honest once a registered custom propellant becomes reachable.
    """
    good = MissionSpec.for_material("GRCop-84", 13.0e3)
    x, surfaces, r = _solved(good)
    mission = replace(good, propellant_name="LOX/Unobtainium")

    inputs = resolve_engine_inputs(
        x.as_contract_dict(effective_of=r.OF),
        mission,
        effective_of=float(r.OF),
        of_source="mission_nominal",
        total_mass_flow=float(r.mdot),
        surfaces=surfaces,
    )
    assert inputs.propellant.oxidizer == "LOX"
    assert inputs.propellant.fuel == "Unobtainium"
    assert inputs.propellant.coolant == "Unobtainium"
    assert "coolant_identity" in inputs.unavailable
    assert inputs.model_identities["coolant_identity_basis"] == (
        "inferred_from_regenerative_architecture"
    )

    catalogued = _resolve(MissionSpec.for_material("GRCop-84", 13.0e3))
    assert catalogued.propellant.coolant == "RP-1"
    assert catalogued.model_identities["coolant_identity_basis"] == "catalog"
    assert "coolant_identity" not in catalogued.unavailable


def test_efficiency_convention_is_stated_once():
    inputs = _resolve(MissionSpec.for_material("GRCop-84", 13.0e3))
    perf = inputs.performance
    assert perf.eta_Isp == pytest.approx(perf.eta_cstar_effective * perf.eta_CF)
    assert perf.delivered_convention == "eta_cstar_times_eta_CF"


def test_digest_is_content_addressed_and_changes_with_a_real_input_change():
    a = _resolve(MissionSpec.for_material("GRCop-84", 13.0e3))
    b = _resolve(MissionSpec.for_material("GRCop-84", 13.0e3))
    c = _resolve(MissionSpec.for_material("NARloy-Z", 13.0e3))
    assert a.digest() == b.digest()
    assert a.digest() != c.digest()


def test_property_table_identity_is_carried_for_parity_proof():
    inputs = _resolve(MissionSpec.for_material("GRCop-84", 13.0e3))
    assert inputs.model_identities["property_table_sha256"]
    assert inputs.model_identities["contour_provider"]


def test_postsolve_contour_evidence_is_content_addressed_and_cannot_orphan():
    mission = MissionSpec.for_material("GRCop-84", 13.0e3)
    policy = {
        "policy_id": "top-vs-variational-screening-tolerances-v1",
        "max_radius_error_over_rt": 0.05,
        "rms_radius_error_over_rt": 0.02,
        "relative_divergent_length_error": 0.02,
        "relative_thrust_coefficient_error": 0.03,
    }
    inputs = _resolve(
        mission,
        postsolve_contour_validator_id="top_vs_rao_variational_postsolve_v1",
        rao_convergence_domain_sha256="a" * 64,
        contour_comparison_policy=policy,
    )

    assert inputs.model_identities["postsolve_contour_validator"] == (
        "top_vs_rao_variational_postsolve_v1"
    )
    assert inputs.model_identities["rao_convergence_domain_sha256"] == "a" * 64
    assert inputs.model_identities["contour_comparison_policy"] == policy

    with pytest.raises(ValueError, match="require a selected"):
        _resolve(
            mission,
            rao_convergence_domain_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="SHA-256"):
        _resolve(
            mission,
            postsolve_contour_validator_id="top_vs_rao_variational_postsolve_v1",
            rao_convergence_domain_sha256="not-a-hash",
            contour_comparison_policy=policy,
        )


@pytest.mark.parametrize("liner", ["GRCop-84", "NARloy-Z", "CuCrZr"])
def test_traditional_bridge_carries_exactly_the_resolved_conventions(liner):
    """Anti-drift gate for remediation item 9.

    Note what this does and does not prove now that the switch is done.
    ``to_design_input`` *consumes* the contract, so for a correctly wired
    field this compares a value with itself. It is therefore no longer
    evidence of two independent derivations agreeing -- that was its role
    during the switch, and it held (zero drift) at that point.

    What it still catches, and why it stays: a mis-wired field (oxidizer tank
    pressure landing on the fuel line reads the wrong contract slot and
    diverges), and any future reintroduction of a direct ``mission.``/design
    read in place of a contract read. Both are live failure modes.
    """
    from raosim.mdo.postprocess import to_design_input
    from raosim.mdo.resolved_inputs import crosscheck_design_input

    mission = MissionSpec.for_material(liner, 13.0e3)
    x, surfaces, r = _solved(mission)
    design = x.as_contract_dict(effective_of=r.OF)

    resolved = resolve_engine_inputs(
        design, mission, effective_of=float(r.OF),
        of_source="mission_nominal", total_mass_flow=float(r.mdot),
        surfaces=surfaces,
    )
    design_input = to_design_input(design, mission, effective_of=float(r.OF))

    drift = crosscheck_design_input(resolved, design_input)
    assert drift == (), "resolved inputs and DesignInput disagree:\n  " + \
        "\n  ".join(drift)


def test_crosscheck_actually_detects_a_planted_divergence():
    """A gate that cannot fail proves nothing."""
    from dataclasses import replace as _replace

    from raosim.mdo.postprocess import to_design_input
    from raosim.mdo.resolved_inputs import crosscheck_design_input

    mission = MissionSpec.for_material("GRCop-84", 13.0e3)
    x, surfaces, r = _solved(mission)
    design = x.as_contract_dict(effective_of=r.OF)
    resolved = resolve_engine_inputs(
        design, mission, effective_of=float(r.OF),
        of_source="mission_nominal", total_mass_flow=float(r.mdot),
        surfaces=surfaces,
    )
    design_input = to_design_input(design, mission, effective_of=float(r.OF))
    design_input.material.conductivity = float(
        design_input.material.conductivity
    ) * 1.05

    drift = crosscheck_design_input(resolved, design_input)
    assert any("liner_conductivity" in row for row in drift)


def test_bridge_derives_physics_only_from_the_contract():
    """``to_design_input`` must stay a consumer, not drift back to a builder.

    Once the contract is resolved, no physical value may be re-read from the
    mission or the raw design dict -- that is precisely how two definitions of
    one convention reappear. The single tolerated read is the thermochemistry
    *backend* selection, which is analysis configuration rather than a
    physical input and is recorded in ``model_identities``.
    """
    import inspect
    import re

    from raosim.mdo import postprocess

    source = inspect.getsource(postprocess.to_design_input)
    _, _, body = source.partition("resolved.assert_flow_closure()")
    assert body, "the contract-resolution anchor moved; update this guard"

    offenders = [
        line.strip()
        for line in body.splitlines()
        if re.search(r"\bmission\.|d\[\"", line)
    ]
    allowed = {
        'else ("cea_frozen" if mission.cea_table_path else "constant_gamma")'
    }
    unexpected = [line for line in offenders if line not in allowed]
    assert not unexpected, (
        "to_design_input re-derives physics instead of reading the resolved "
        "contract:\n  " + "\n  ".join(unexpected)
    )


def test_parity_run_crosschecks_the_handoff_and_records_the_identity():
    """The contract must be load-bearing at run time, not only in tests.

    A real ``reevaluate`` has to resolve the contract, crosscheck the
    traditional handoff against it, and record the content-addressed identity
    so a parity claim can be checked against the inputs rather than assumed.
    """
    from raosim.mdo.postprocess import reevaluate
    from raosim.mdo.state import solve_engine_state

    mission = MissionSpec.for_material("GRCop-84", 13.0e3)
    x, surfaces, _ = _solved(mission)
    state = solve_engine_state(x, mission, surfaces=surfaces)
    design = {k: float(v) for k, v in x.as_dict().items()}

    reev = reevaluate(
        design, mission, mdo_result=state, mdo_surfaces=surfaces,
        mdo_summary={"Isp": float(state.performance.Isp_delivered)},
        size_pumps=False,
    )

    metadata = reev.authoritative_snapshot.optimizer_metadata or {}
    recorded = metadata.get("resolved_engine_inputs")
    assert recorded is not None, "parity run did not record the input contract"
    assert recorded["traditional_handoff_crosscheck"] == "agrees"
    assert recorded["schema_version"] == RESOLVED_INPUTS_SCHEMA_VERSION
    assert len(recorded["digest"]) == 64
    assert recorded["liner_material"] == "GRCop-84"
    assert recorded["closeout_material"] == "Inconel 718"

    assert not [
        w for w in reev.authoritative_snapshot.warnings if "disagree" in w.lower()
    ]


def test_nonphysical_inputs_are_rejected_at_construction():
    mission = MissionSpec.for_material("GRCop-84", 13.0e3)
    with pytest.raises(ValueError, match="effective_of must be positive"):
        _resolve(mission, effective_of=0.0)
    with pytest.raises(ValueError, match="must be positive"):
        _resolve(mission, total_mass_flow=-1.0)
