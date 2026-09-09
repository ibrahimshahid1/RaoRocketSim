"""The versioned engine-job contract every front end compiles into.

The gates here encode the two structural decisions: design intent is always
explicit (never inferred from whether a value is present), and a job
*contains* the physical input contract rather than restating it.
"""

from __future__ import annotations

import pytest

from raosim.mdo.job import (
    ArchitectureSelection,
    ConstraintPolicy,
    EngineJobSpec,
    JobValidationError,
    MissionRequest,
    ModelSelection,
    ObjectiveRequest,
    OperatingPointSpec,
    TargetRequest,
    VariableRequest,
    resolve_engine_job,
)

_FIXED_POINT = (
    ("Pc", "3 MPa"), ("eps", 8.0), ("dp_f_frac", 0.2), ("dp_o_frac", 0.2),
    ("D_pintle", "20 mm"), ("N_rpm", "30 krpm"), ("channel_width", "0.5 mm"),
    ("channel_height", "1.5 mm"), ("film_frac", 0.1), ("t_wall", "0.8 mm"),
)


def _analyze_job(**kw):
    params = dict(
        mission=MissionRequest(thrust="13 kN", burn_duration="120 s"),
        workflow="analyze",
        architecture=ArchitectureSelection(liner_material="GRCop-84"),
        variables=tuple(
            VariableRequest(n, "fixed", v) for n, v in _FIXED_POINT
        ),
    )
    params.update(kw)
    return EngineJobSpec(**params)


def test_analyze_job_resolves_to_a_complete_fixed_design_point():
    job = resolve_engine_job(_analyze_job())
    assert job.active_variables == ()
    assert job.fixed_variables["Pc"] == pytest.approx(3.0e6)
    assert job.fixed_variables["channel_width"] == pytest.approx(0.5e-3)
    assert job.mission.liner_material_name == "GRCop-84"
    assert len(job.job_digest) == 64


def test_job_contains_the_resolved_input_contract_rather_than_restating_it():
    """The sequencing decision, made concrete and executable."""
    from raosim.mdo.engine import chamber_surfaces_for, solve_engine
    from raosim.mdo.resolved_inputs import resolve_engine_inputs

    job = resolve_engine_job(_analyze_job())
    x = job.design_vector()
    surfaces = chamber_surfaces_for(job.mission)
    r = solve_engine(x, job.mission, surfaces=surfaces)
    inputs = resolve_engine_inputs(
        x.as_contract_dict(effective_of=r.OF), job.mission,
        effective_of=float(r.OF), of_source="mission_nominal",
        total_mass_flow=float(r.mdot), surfaces=surfaces,
    )
    job = job.with_inputs(inputs)

    payload = job.to_dict()
    assert payload["resolved_inputs"] is not None
    assert payload["job_digest"] == job.job_digest
    # One definition of the material, reachable through the job.
    assert job.inputs.material.liner_name == job.mission.liner_material_name
    assert job.inputs.target_thrust == pytest.approx(13.0e3)


def test_a_design_vector_is_refused_while_variables_are_still_free():
    spec = _analyze_job(
        workflow="optimize",
        objective=ObjectiveRequest("masses.dry_mass_partial", "min"),
        variables=(VariableRequest("Pc", "free"),),
    )
    job = resolve_engine_job(spec)
    with pytest.raises(JobValidationError, match="every variable is fixed"):
        job.design_vector()


def test_unstated_analyze_variables_are_named_as_defaults_not_choices():
    """A value the user did not choose must not be reported as one."""
    spec = _analyze_job(variables=(VariableRequest("Pc", "fixed", "3 MPa"),))
    job = resolve_engine_job(spec)

    assert job.fixed_variables["Pc"] == pytest.approx(3.0e6)
    note = next(n for n in job.notes if "reference value" in n)
    for name in ("eps", "t_wall", "channel_width", "film_frac"):
        assert name in note
    assert "Pc" not in note.split(":", 1)[1].split(".")[0]


def test_a_fully_stated_analyze_job_reports_no_defaults():
    job = resolve_engine_job(_analyze_job())
    assert not [n for n in job.notes if "reference value" in n]


def test_units_are_required_on_dimensional_variables():
    with pytest.raises(JobValidationError, match="ambiguous"):
        VariableRequest("Pc", "fixed", 3.0e6)
    with pytest.raises(JobValidationError, match="ambiguous"):
        VariableRequest("channel_width", "bounded", lower=0.4e-3, upper="1 mm")


def test_variable_intent_is_explicit_not_inferred():
    with pytest.raises(JobValidationError, match="requires an explicit value"):
        VariableRequest("Pc", "fixed")
    with pytest.raises(JobValidationError, match="requires both lower and upper"):
        VariableRequest("Pc", "bounded", lower="2 MPa")
    with pytest.raises(JobValidationError, match="at least two values"):
        VariableRequest("channel_width", "enumerate", values=("0.5 mm",))
    with pytest.raises(JobValidationError, match="must be below"):
        VariableRequest("Pc", "bounded", lower="8 MPa", upper="2 MPa")


def test_derived_mode_needs_a_documented_resolver():
    """``derived`` must not become a synonym for free."""
    with pytest.raises(JobValidationError, match="documented resolver"):
        VariableRequest("Pc", "derived")
    assert VariableRequest("N_rpm", "derived").name == "N_rpm"


def test_unknown_variable_names_are_rejected():
    with pytest.raises(JobValidationError, match="unknown design variable"):
        VariableRequest("chamber_pressure", "fixed", "3 MPa")


def test_bounds_beyond_the_validated_domain_are_clipped_with_a_note():
    spec = _analyze_job(
        workflow="optimize",
        objective=ObjectiveRequest("masses.dry_mass_partial", "min"),
        variables=(VariableRequest("Pc", "bounded", lower="2 MPa", upper="8 MPa"),),
    )
    job = resolve_engine_job(spec)
    lo, hi = job.bounds["Pc"]
    assert lo == pytest.approx(2.0e6)
    assert hi == pytest.approx(6.0e6)
    assert any("beyond the configured search domain" in n for n in job.notes)


def test_optimize_requires_an_objective_and_analyze_forbids_one():
    with pytest.raises(JobValidationError, match="requires an objective"):
        _analyze_job(workflow="optimize", variables=())
    with pytest.raises(JobValidationError, match="cannot carry"):
        _analyze_job(objective=ObjectiveRequest("masses.dry_mass_partial"))


def test_of_optimization_requires_a_validated_property_table():
    """O/F is a lever only when the thermochemistry actually depends on it."""
    with pytest.raises(JobValidationError, match="requires a validated"):
        ModelSelection(mixture_ratio_mode="optimize")
    with pytest.raises(JobValidationError, match="mixture_ratio_mode"):
        _analyze_job(variables=(VariableRequest("OF", "free"),))


def test_postsolve_contour_validation_is_an_exact_explicit_model_selection():
    from raosim.mdo.contour_validation import VARIATIONAL_VALIDATOR_MODEL_ID

    with pytest.raises(JobValidationError, match="unknown post-solve"):
        ModelSelection(
            postsolve_contour_validator="variational-ish",
            rao_convergence_domain="domain.json",
            contour_max_radius_error_over_rt=0.05,
            contour_rms_radius_error_over_rt=0.02,
            contour_relative_length_error=0.02,
            contour_relative_cf_error=0.03,
        )
    with pytest.raises(JobValidationError, match="without postsolve"):
        ModelSelection(rao_convergence_domain="domain.json")
    with pytest.raises(JobValidationError, match="explicit discrepancy"):
        ModelSelection(
            postsolve_contour_validator=VARIATIONAL_VALIDATOR_MODEL_ID,
            rao_convergence_domain="domain.json",
        )
    with pytest.raises(JobValidationError, match="not bool"):
        ModelSelection(
            postsolve_contour_validator=VARIATIONAL_VALIDATOR_MODEL_ID,
            rao_convergence_domain="domain.json",
            contour_max_radius_error_over_rt=True,
            contour_rms_radius_error_over_rt=0.02,
            contour_relative_length_error=0.02,
            contour_relative_cf_error=0.03,
        )
    selected = ModelSelection(
        postsolve_contour_validator=VARIATIONAL_VALIDATOR_MODEL_ID,
        rao_convergence_domain="domain.json",
        contour_max_radius_error_over_rt=0.05,
        contour_rms_radius_error_over_rt=0.02,
        contour_relative_length_error=0.02,
        contour_relative_cf_error=0.03,
    )
    assert selected.postsolve_contour_validator == VARIATIONAL_VALIDATOR_MODEL_ID
    changed_policy = ModelSelection(
        postsolve_contour_validator=VARIATIONAL_VALIDATOR_MODEL_ID,
        rao_convergence_domain="domain.json",
        contour_max_radius_error_over_rt=0.06,
        contour_rms_radius_error_over_rt=0.02,
        contour_relative_length_error=0.02,
        contour_relative_cf_error=0.03,
    )
    assert _analyze_job(models=selected).digest() != _analyze_job(
        models=changed_policy
    ).digest()


def test_job_resolution_eagerly_validates_the_selected_rao_domain(tmp_path):
    from raosim.mdo.contour_validation import VARIATIONAL_VALIDATOR_MODEL_ID

    models = ModelSelection(
        postsolve_contour_validator=VARIATIONAL_VALIDATOR_MODEL_ID,
        rao_convergence_domain=str(tmp_path / "missing.json"),
        contour_max_radius_error_over_rt=0.05,
        contour_rms_radius_error_over_rt=0.02,
        contour_relative_length_error=0.02,
        contour_relative_cf_error=0.03,
    )
    with pytest.raises(JobValidationError, match="artifact is invalid"):
        resolve_engine_job(_analyze_job(models=models))


def test_resolved_job_digest_is_bound_to_the_domain_content_hash(monkeypatch):
    from raosim.mdo.contour_validation import VARIATIONAL_VALIDATOR_MODEL_ID
    from raosim.rao_existence_scan import RaoConvergenceDomain

    selected_hash = ["a" * 64]

    class SyntheticDomain:
        @property
        def artifact_sha256(self):
            return selected_hash[0]

    monkeypatch.setattr(
        RaoConvergenceDomain,
        "read_json",
        classmethod(lambda cls, path: SyntheticDomain()),
    )
    models = ModelSelection(
        postsolve_contour_validator=VARIATIONAL_VALIDATOR_MODEL_ID,
        rao_convergence_domain="domain.json",
        contour_max_radius_error_over_rt=0.05,
        contour_rms_radius_error_over_rt=0.02,
        contour_relative_length_error=0.02,
        contour_relative_cf_error=0.03,
    )
    first = resolve_engine_job(_analyze_job(models=models))
    selected_hash[0] = "b" * 64
    second = resolve_engine_job(_analyze_job(models=models))

    assert first.spec.digest() == second.spec.digest()
    assert first.job_digest != second.job_digest
    assert first.rao_convergence_domain_sha256 == "a" * 64
    assert second.rao_convergence_domain_sha256 == "b" * 64


def test_duplicate_variable_policies_are_rejected():
    with pytest.raises(JobValidationError, match="only one policy"):
        _analyze_job(variables=(
            VariableRequest("Pc", "fixed", "3 MPa"),
            VariableRequest("Pc", "free"),
        ))


def test_multipoint_is_refused_rather_than_silently_single_point():
    """A throttle envelope must not come back as a single-point answer."""
    with pytest.raises(JobValidationError, match="multipoint operation is not"):
        _analyze_job(operating_points=(
            OperatingPointSpec("sl", "101325 Pa"),
            OperatingPointSpec("vac", "0 Pa"),
        ))


def test_qualification_duration_cannot_undercut_flight_duration():
    with pytest.raises(JobValidationError, match="cannot be below"):
        MissionRequest(
            thrust="13 kN", burn_duration="120 s",
            qualification_duration="60 s",
        ).resolved()


def test_altitude_condition_requires_an_altitude():
    with pytest.raises(JobValidationError, match="requires an altitude"):
        MissionRequest(thrust="13 kN", thrust_condition="altitude").resolved()
    resolved = MissionRequest(
        thrust="13 kN", thrust_condition="altitude", altitude="10 km"
    ).resolved()
    assert resolved["altitude"] == pytest.approx(10.0e3)


def test_ablation_policy_cannot_present_itself_as_physics_feasible():
    with pytest.raises(JobValidationError, match="no constraint-selection mask"):
        ModelSelection(constraint_policy=ConstraintPolicy.CUSTOM_ABLATION)


def test_missing_material_selection_is_reported_not_assumed():
    spec = _analyze_job(architecture=ArchitectureSelection())
    job = resolve_engine_job(spec)
    assert job.mission.liner_material_name is None
    assert any("no single catalog alloy" in n for n in job.notes)


def test_targets_parse_units_and_validate_ranges():
    target = TargetRequest(
        "performance.isp_delivered", "at_least", "230 s", dimension="time"
    )
    assert target.value == pytest.approx(230.0)
    with pytest.raises(JobValidationError, match="needs a value"):
        TargetRequest("performance.isp_delivered", "at_least")
    with pytest.raises(JobValidationError, match="needs value and upper_value"):
        TargetRequest("performance.isp_delivered", "range", "230 s",
                      dimension="time")


def test_objectives_and_targets_validate_against_the_metric_registry():
    """A job cannot request an output the repository cannot actually produce."""
    with pytest.raises(JobValidationError, match="unknown metric"):
        ObjectiveRequest("masses.everything")
    # "Minimize engine dry mass" must fail loudly rather than silently
    # optimizing a partial subtotal under a whole-engine name.
    with pytest.raises(JobValidationError, match="valves, lines"):
        ObjectiveRequest("masses.engine_dry_mass")
    with pytest.raises(JobValidationError, match="cannot be a primary"):
        ObjectiveRequest("geometry.throat_radius")
    for unsupported in (
        "performance.isp_delivered",
        "feed.electric_power_total",
        "masses.battery",
    ):
        with pytest.raises(JobValidationError, match="cannot be a primary"):
            ObjectiveRequest(unsupported)
    with pytest.raises(JobValidationError, match="minimization-only"):
        ObjectiveRequest("masses.dry_mass_partial", "max")
    with pytest.raises(JobValidationError, match="must not be re-targeted"):
        TargetRequest("constraints.wall_temp_margin", "at_least", 0.0)


def test_target_units_come_from_the_registry_not_the_caller():
    target = TargetRequest("performance.isp_delivered", "at_least", "230 s")
    assert target.dimension == "time"
    assert target.value == pytest.approx(230.0)
    with pytest.raises(JobValidationError, match="registered dimension"):
        TargetRequest(
            "performance.isp_delivered", "at_least", "230 s",
            dimension="pressure",
        )
    # A quantity in the wrong unit for its registered dimension is rejected.
    with pytest.raises(JobValidationError, match="is a force unit"):
        TargetRequest("performance.isp_delivered", "at_least", "230 kN")


def test_volume_and_heat_flux_targets_parse_unit_bearing_values():
    chamber_volume = TargetRequest(
        "geometry.chamber_volume", "at_most", "2.5 L"
    )
    assert chamber_volume.dimension == "volume"
    assert chamber_volume.value == pytest.approx(2.5e-3)

    heat_flux = TargetRequest(
        "thermal.heat_flux_max", "at_most", "7.5 MW/m^2"
    )
    assert heat_flux.dimension == "heat_flux"
    assert heat_flux.value == pytest.approx(7.5e6)


def test_the_same_job_digests_identically_and_a_changed_one_does_not():
    a = resolve_engine_job(_analyze_job()).job_digest
    b = resolve_engine_job(_analyze_job()).job_digest
    c = resolve_engine_job(
        _analyze_job(mission=MissionRequest(thrust="20 kN"))
    ).job_digest
    assert a == b
    assert a != c
