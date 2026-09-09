"""One entry point, four-valued verdicts.

A user should not have to know whether an answer came from the differentiable
core or the traditional pipeline. These gates pin the parts of that promise
that are easy to erode: the verdict must be able to say "unknown", a failed
solve must not be reported as a violated requirement, and a violation must name
itself.
"""

from __future__ import annotations

import pytest

from raosim.input_validation import WorkflowExitCode
from raosim.mdo.execute import TargetOutcome, _verdict, execute_engine_job
from raosim.mdo.job import (
    ArchitectureSelection,
    EngineJobSpec,
    JobValidationError,
    MissionRequest,
    ModelSelection,
    ObjectiveRequest,
    TargetRequest,
    VariableRequest,
)

_POINT = (
    ("Pc", "3 MPa"), ("eps", 8.0), ("dp_f_frac", 0.2), ("dp_o_frac", 0.2),
    ("D_pintle", "20 mm"), ("N_rpm", "30 krpm"), ("channel_width", "0.5 mm"),
    ("channel_height", "1.5 mm"), ("film_frac", 0.1), ("t_wall", "0.8 mm"),
)


def _spec(**kw):
    params = dict(
        mission=MissionRequest(thrust="13 kN", burn_duration="120 s"),
        workflow="analyze",
        architecture=ArchitectureSelection(liner_material="GRCop-84"),
        variables=tuple(VariableRequest(n, "fixed", v) for n, v in _POINT),
    )
    params.update(kw)
    return EngineJobSpec(**params)


@pytest.fixture(scope="module")
def analyzed():
    return execute_engine_job(_spec(
        targets=(TargetRequest("performance.isp_delivered", "at_least", "230 s"),)
    ))


def test_one_call_produces_verdict_inputs_and_snapshot(analyzed):
    from raosim.mdo.nozzle import (
        ANALYTIC_TOP_MODEL_CODE,
        ANALYTIC_TOP_PROVIDER_ID,
    )

    assert analyzed.verdict in {
        "MET", "CANDIDATE_VIOLATES", "INDETERMINATE", "SOLVER_FAILED"
    }
    assert analyzed.snapshot is not None
    assert analyzed.state is not None
    assert analyzed.inputs is not None
    assert len(analyzed.inputs.digest()) == 64
    assert analyzed.inputs.material.liner_name == "GRCop-84"
    assert analyzed.inputs.contour.provider_id == ANALYTIC_TOP_PROVIDER_ID
    assert (
        analyzed.inputs.model_identities["contour_provider"]
        == ANALYTIC_TOP_PROVIDER_ID
    )
    assert int(analyzed.state.geometry.divergent_model) == ANALYTIC_TOP_MODEL_CODE


def test_targets_are_evaluated_against_the_authoritative_snapshot(analyzed):
    isp = next(
        t for t in analyzed.targets
        if t.metric == "performance.isp_delivered"
    )
    assert isp.achieved is not None
    assert isp.status == ("pass" if isp.achieved >= 230.0 else "fail")


def test_a_violation_names_itself_with_its_margin(analyzed):
    """A verdict that only says 'violates' makes the user go hunting."""
    if analyzed.verdict != "CANDIDATE_VIOLATES":
        pytest.skip("reference point is not violating in this configuration")
    assert analyzed.violated_constraints
    for name, margin in analyzed.violated_constraints:
        assert isinstance(name, str) and name
        assert margin < 0.0
    # Worst first.
    margins = [m for _, m in analyzed.violated_constraints]
    assert margins == sorted(margins)


def test_result_serializes_with_both_digests(analyzed):
    payload = analyzed.to_dict()
    assert payload["job_digest"] == analyzed.job.job_digest
    assert payload["resolved_inputs_digest"] == analyzed.inputs.digest()
    assert payload["exit_code"] == analyzed.exit_code
    assert isinstance(payload["violated_constraints"], list)


def test_default_job_never_invokes_the_host_variational_solver(monkeypatch):
    import raosim.mdo.contour_validation as validation_module

    calls = 0

    def forbidden(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("default analytic job invoked the host validator")

    monkeypatch.setattr(
        validation_module,
        "validate_optimized_top_contour",
        forbidden,
    )

    result = execute_engine_job(_spec())

    assert calls == 0
    assert result.contour_validation is None
    assert "contour_validation" not in result.statuses


def test_selected_host_validator_runs_once_and_persists_typed_unknown(
    monkeypatch,
):
    import raosim.mdo.contour_validation as validation_module
    from raosim.mdo.contour_validation import (
        VARIATIONAL_VALIDATOR_MODEL_ID,
        VariationalValidationReason,
        unavailable_variational_validation,
    )
    from raosim.rao_existence_scan import RaoConvergenceDomain

    domain_hash = "b" * 64

    class SyntheticDomain:
        artifact_sha256 = domain_hash

    monkeypatch.setattr(
        RaoConvergenceDomain,
        "read_json",
        classmethod(lambda cls, path: SyntheticDomain()),
    )
    calls = 0

    def typed_unknown(state, mission, config):
        nonlocal calls
        calls += 1
        assert config.convergence_domain.artifact_sha256 == domain_hash
        return unavailable_variational_validation(
            "synthetic fixed-domain preflight is under-resolved",
            reason=VariationalValidationReason.CONVERGENCE_DOMAIN_UNSUPPORTED,
        )

    monkeypatch.setattr(
        validation_module,
        "validate_optimized_top_contour",
        typed_unknown,
    )
    models = ModelSelection(
        postsolve_contour_validator=VARIATIONAL_VALIDATOR_MODEL_ID,
        rao_convergence_domain="synthetic-domain.json",
        contour_max_radius_error_over_rt=0.05,
        contour_rms_radius_error_over_rt=0.02,
        contour_relative_length_error=0.02,
        contour_relative_cf_error=0.03,
    )

    result = execute_engine_job(_spec(models=models))

    assert calls == 1
    assert result.contour_validation is not None
    assert result.contour_validation.status.value == "unknown"
    assert result.statuses["contour_validation"] == "unknown"
    assert result.verdict != "MET"
    assert result.inputs.model_identities[
        "postsolve_contour_validator"
    ] == VARIATIONAL_VALIDATOR_MODEL_ID
    assert result.inputs.model_identities[
        "rao_convergence_domain_sha256"
    ] == domain_hash
    assert result.job.rao_convergence_domain_sha256 == domain_hash
    for snapshot in (
        result.reevaluation.mdo_snapshot,
        result.reevaluation.authoritative_snapshot,
    ):
        persisted = snapshot.optimizer_metadata[
            "variational_contour_validation"
        ]
        assert persisted["status"] == "unknown"
        assert persisted["reason"] == "convergence_domain_unsupported"


def test_domain_identity_change_between_resolution_and_execution_is_refused(
    monkeypatch,
):
    from raosim.mdo.contour_validation import VARIATIONAL_VALIDATOR_MODEL_ID
    from raosim.rao_existence_scan import RaoConvergenceDomain

    hashes = iter(("d" * 64, "e" * 64))

    class SyntheticDomain:
        def __init__(self, digest):
            self.artifact_sha256 = digest

    monkeypatch.setattr(
        RaoConvergenceDomain,
        "read_json",
        classmethod(lambda cls, path: SyntheticDomain(next(hashes))),
    )
    models = ModelSelection(
        postsolve_contour_validator=VARIATIONAL_VALIDATOR_MODEL_ID,
        rao_convergence_domain="mutable-domain.json",
        contour_max_radius_error_over_rt=0.05,
        contour_rms_radius_error_over_rt=0.02,
        contour_relative_length_error=0.02,
        contour_relative_cf_error=0.03,
    )

    with pytest.raises(JobValidationError, match="changed after job resolution"):
        execute_engine_job(_spec(models=models))


def test_exit_codes_follow_the_shared_workflow_contract(analyzed):
    assert analyzed.exit_code == {
        "MET": WorkflowExitCode.MET,
        "CANDIDATE_VIOLATES": WorkflowExitCode.CANDIDATE_VIOLATES,
        "INDETERMINATE": WorkflowExitCode.INDETERMINATE,
        "SOLVER_FAILED": WorkflowExitCode.SOLVER_FAILED,
    }[analyzed.verdict]


# --------------------------------------------------------------------------- #
# Verdict reduction: the part where an unknown could silently become a pass.   #
# --------------------------------------------------------------------------- #
def test_a_failed_solve_is_not_reported_as_a_violated_requirement():
    """A failed nonconvex solve is not proof a requirement is impossible."""
    verdict, code = _verdict({"workflow": "fail", "physics": "fail"}, ())
    assert verdict == "SOLVER_FAILED"
    assert code == WorkflowExitCode.SOLVER_FAILED

    verdict, _ = _verdict({"workflow": "pass", "numerical": "fail"}, ())
    assert verdict == "SOLVER_FAILED"


def test_unknown_never_becomes_met():
    verdict, code = _verdict(
        {"workflow": "pass", "numerical": "pass", "physics": "unknown"}, ()
    )
    assert verdict == "INDETERMINATE"
    assert code == WorkflowExitCode.INDETERMINATE


def test_a_demonstrated_failure_outranks_an_unknown():
    verdict, _ = _verdict(
        {"workflow": "pass", "numerical": "pass",
         "physics": "fail", "requirements": "unknown"}, ()
    )
    assert verdict == "CANDIDATE_VIOLATES"


def test_postsolve_contour_gate_preserves_fail_and_unknown_semantics():
    failed, failed_code = _verdict(
        {
            "workflow": "pass",
            "numerical": "pass",
            "physics": "pass",
            "requirements": "pass",
            "contour_validation": "fail",
        },
        (),
    )
    assert failed == "CANDIDATE_VIOLATES"
    assert failed_code == WorkflowExitCode.CANDIDATE_VIOLATES

    unknown, unknown_code = _verdict(
        {
            "workflow": "pass",
            "numerical": "pass",
            "physics": "pass",
            "requirements": "pass",
            "contour_validation": "unknown",
        },
        (),
    )
    assert unknown == "INDETERMINATE"
    assert unknown_code == WorkflowExitCode.INDETERMINATE


def test_a_missed_target_violates_even_when_every_gate_passes():
    missed = TargetOutcome(
        "performance.isp_delivered", "at_least", 400.0, None, 230.0, "fail"
    )
    verdict, _ = _verdict(
        {"workflow": "pass", "numerical": "pass", "physics": "pass",
         "requirements": "pass", "optimizer": "pass"},
        (missed,),
    )
    assert verdict == "CANDIDATE_VIOLATES"


def test_an_unknown_target_is_indeterminate_not_met():
    unknown = TargetOutcome(
        "masses.thrust_chamber", "at_most", 5.0, None, None, "unknown",
        "mass branch unavailable",
    )
    verdict, _ = _verdict(
        {"workflow": "pass", "numerical": "pass", "physics": "pass",
         "requirements": "pass", "optimizer": "pass"},
        (unknown,),
    )
    assert verdict == "INDETERMINATE"


def test_all_pass_is_met():
    from raosim.mdo.execute import _REQUIRED_STATUS_KEYS

    verdict, code = _verdict(
        {name: "pass" for name in _REQUIRED_STATUS_KEYS}, ()
    )
    assert verdict == "MET"
    assert code == WorkflowExitCode.MET


def test_pareto_is_refused_rather_than_returning_one_point():
    with pytest.raises(JobValidationError, match="not wired through"):
        execute_engine_job(_spec(
            workflow="pareto",
            objective=ObjectiveRequest("masses.dry_mass_partial", "min"),
        ))


# --------------------------------------------------------------------------- #
# Discrete architecture enumeration                                            #
# --------------------------------------------------------------------------- #
def test_list_valued_selections_expand_to_one_job_each():
    from raosim.mdo.execute import expand_architectures

    spec = _spec(architecture=ArchitectureSelection(
        liner_material=["GRCop-84", "NARloy-Z", "CuCrZr"],
    ))
    expanded = expand_architectures(spec)
    assert len(expanded) == 3
    assert [s.architecture.liner_material for s in expanded] == [
        "GRCop-84", "NARloy-Z", "CuCrZr"
    ]
    # Discrete choices stay out of the gradient: each is its own problem.
    for candidate in expanded:
        assert isinstance(candidate.architecture.liner_material, str)


def test_a_scalar_selection_still_yields_exactly_one_job():
    from raosim.mdo.execute import expand_architectures

    assert len(expand_architectures(_spec())) == 1


@pytest.mark.parametrize("liners", [["GRCop-84", "OFHC Copper"]])
def test_material_comparison_differentiates_the_traced_physics(liners):
    """The payoff of the atomic material mapper.

    Different alloys must produce different constraint outcomes; identical
    results would mean the selection never reached the traced core.
    """
    from raosim.mdo.execute import execute_engine_job_set

    outcome = execute_engine_job_set(_spec(
        architecture=ArchitectureSelection(liner_material=liners)
    ))
    assert len(outcome.results) == len(liners)

    by_liner = {
        r.job.spec.architecture.liner_material:
            {name for name, _ in r.violated_constraints}
        for r in outcome.results
    }
    # OFHC Copper is the weakest catalog liner (70 MPa yield) and must show a
    # structural failure that GRCop-84 (186 MPa) does not.
    assert "structural_stress" in by_liner["OFHC Copper"]
    assert "structural_stress" not in by_liner["GRCop-84"]


def test_only_met_candidates_are_ranked():
    """Unknown coverage is not 'slightly worse' than a requirement met."""
    from raosim.mdo.execute import execute_engine_job_set

    outcome = execute_engine_job_set(_spec(
        architecture=ArchitectureSelection(
            liner_material=["GRCop-84", "NARloy-Z"]
        )
    ))
    for result in outcome.ranked:
        assert result.verdict == "MET"
    for result in outcome.unranked:
        assert result.verdict != "MET"
    assert len(outcome.ranked) + len(outcome.unranked) == len(outcome.results)
    if not outcome.ranked:
        assert outcome.best is None
    # Every candidate remains reportable even when none is rankable.
    payload = outcome.to_dict()
    assert len(payload["ranked"]) + len(payload["unranked"]) == 2
    for row in payload["unranked"]:
        assert row["verdict"] and "liner_material" in row
