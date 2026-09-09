import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

import raosim.rao_existence_scan as scan_module
from raosim.rao_variational import RAO_RADIAL_INCREMENT_SOURCE_ID
from raosim.rao_existence_scan import (
    CONVERGENCE_DOMAIN_SOURCE_MODEL_ID,
    DOMAIN_CONTEXT_MISMATCH,
    DOMAIN_REFUSED_INVALID_RAO_REGION,
    DOMAIN_SUPPORTED,
    DOMAIN_UNKNOWN_OUTSIDE,
    DOMAIN_UNKNOWN_UNDER_RESOLVED,
    MODEL_SMOOTH,
    RESOLUTION_FAILED,
    RESOLUTION_PASSED,
    SAMPLE_CONVERGED_INVALID,
    SAMPLE_CONVERGED_VALID,
    SAMPLE_NUMERICAL_FAILURE,
    SAMPLE_UNDER_RESOLVED,
    ExistenceScanConfig,
    RaoConvergenceContext,
    RaoConvergenceDomain,
    RaoConvergenceDomainError,
    RaoConvergenceSample,
    RaoKernelResolutionEvidence,
    RootRefineResult,
    evaluate_closure_point,
    refine_geometry_root,
    scan_convergence_domain,
)


def _context(*, rt=0.02, gamma=1.4, model=MODEL_SMOOTH):
    return RaoConvergenceContext(
        Rt_reference=rt,
        gamma=gamma,
        pa_over_p0=0.01,
        throat_downstream_radius_factor=0.382,
        throat_upstream_radius_factor=1.5,
        starting_line_method="kliegel_levine",
        kernel_dtheta_limit_deg=0.25,
        kernel_massflow_rel_tol=0.02,
        n_kernel_finest=12,
        n_kernel_ladder=(8, 12),
        n_de_points=16,
        max_de_steps=240,
        root_maxfev=80,
        theta_b_seed_policy="fixed_axis",
        theta_b_seed_axis_deg=(20.0, 30.0),
        theta_b_chart_span_deg=0.0,
        theta_b_chart_count=0,
        theta_b_chart_source_id="",
        kdf_seed_axis=(0.1, 0.5),
        stop_at="mass",
        scan_mode="geometry",
        third_residual="auto",
        residual_weights=(1.0, 1.0, 1.0),
        max_mach=12.0,
        closure_model=model,
    )


def _resolution(status=RESOLUTION_PASSED):
    return RaoKernelResolutionEvidence(
        n_kernel=(8, 12),
        root_success=(True, True),
        theta_b_deg=(25.0, 25.01),
        kdf=(0.20, 0.2002),
        residual_norm_geometry=(2.0e-8, 1.0e-8),
        status=status,
        reason=(
            "final roots agree"
            if status == RESOLUTION_PASSED
            else "final roots do not agree"
        ),
        theta_delta_tolerance_deg=0.05,
        kdf_delta_tolerance=1.0e-3,
    )


def _sample(epsilon, length_pct, classification=SAMPLE_CONVERGED_VALID):
    valid = classification == SAMPLE_CONVERGED_VALID
    under_resolved = classification == SAMPLE_UNDER_RESOLVED
    return RaoConvergenceSample(
        epsilon=epsilon,
        length_pct=length_pct,
        classification=classification,
        reason="synthetic domain evidence",
        theta_b_deg=25.0,
        kdf=0.20,
        residual_norm_geometry=1.0e-8,
        rao_eq5_evaluable=True,
        rao_eq5_valid=valid,
        rao_eq5_min_dr_over_r=0.01 if valid else -0.01,
        resolution=_resolution(
            RESOLUTION_FAILED if under_resolved else RESOLUTION_PASSED
        ),
        source_scan_sha256="a" * 64,
    )


def _domain(*, samples=None, context=None, max_epsilon_spacing=5.0):
    epsilon_axis = (8.0, 12.0)
    length_axis = (70.0, 80.0)
    if samples is None:
        samples = tuple(
            tuple(_sample(epsilon, length) for length in length_axis)
            for epsilon in epsilon_axis
        )
    return RaoConvergenceDomain(
        context=context or _context(),
        epsilon_axis=epsilon_axis,
        length_pct_axis=length_axis,
        samples=samples,
        residual_norm_limit=1.0e-6,
        theta_delta_tolerance_deg=0.05,
        kdf_delta_tolerance=1.0e-3,
        max_epsilon_spacing=max_epsilon_spacing,
        max_length_pct_spacing=10.0,
    )


def _root(n_kernel, theta=25.0, kdf=0.20, success=True):
    return RootRefineResult(
        model=MODEL_SMOOTH,
        success=success,
        message="synthetic root",
        n_kernel=n_kernel,
        theta_B_deg=theta,
        kdf=kdf,
        fan_turn_deg=0.0,
        radius_residual=1.0e-8,
        length_residual=1.0e-8,
        sigma_E_rad=0.0,
        mass_residual=0.0,
        performance_residual=0.0,
        rao_exit_residual=0.0,
        residual_norm_geometry=1.0e-8,
        residual_norm_stationarity=1.0e-8,
        D_M_pre=2.0,
        D_M_post=2.0,
        D_theta_pre_deg=8.0,
        D_theta_post_deg=8.0,
        exit_angle_deg=0.0,
        exit_mach=3.0,
    )


def test_supported_lookup_requires_all_valid_bracketing_samples():
    domain = _domain()

    lookup = domain.lookup(10.0, 75.0, expected_context=domain.context)

    assert lookup.status == DOMAIN_SUPPORTED
    assert lookup.supported
    assert lookup.epsilon_bracket == (8.0, 12.0)
    assert lookup.length_pct_bracket == (70.0, 80.0)
    assert set(lookup.sample_classifications) == {SAMPLE_CONVERGED_VALID}
    assert domain.source_model_id == (
        "raosim.rao_existence_scan:epsilon-length-fixed-context-"
        "d-attachment-domain:v2"
    )
    assert domain.source_model_id == CONVERGENCE_DOMAIN_SOURCE_MODEL_ID
    assert domain.validity_source_id == RAO_RADIAL_INCREMENT_SOURCE_ID
    assert domain.classification_counts()[SAMPLE_CONVERGED_VALID] == 4


def test_invalid_exact_sample_is_refused_with_typed_exception():
    rows = (
        (
            _sample(8.0, 70.0, SAMPLE_CONVERGED_INVALID),
            _sample(8.0, 80.0),
        ),
        (_sample(12.0, 70.0), _sample(12.0, 80.0)),
    )
    domain = _domain(samples=rows)

    lookup = domain.lookup(8.0, 70.0, expected_context=domain.context)

    assert lookup.status == DOMAIN_REFUSED_INVALID_RAO_REGION
    with pytest.raises(RaoConvergenceDomainError) as exc_info:
        domain.require_supported(8.0, 70.0, expected_context=domain.context)
    assert exc_info.value.lookup == lookup


def test_mixed_or_spatially_coarse_cell_is_explicitly_under_resolved():
    mixed_rows = (
        (
            _sample(8.0, 70.0, SAMPLE_CONVERGED_INVALID),
            _sample(8.0, 80.0),
        ),
        (_sample(12.0, 70.0), _sample(12.0, 80.0)),
    )
    mixed = _domain(samples=mixed_rows)
    coarse = _domain(max_epsilon_spacing=3.0)

    assert mixed.lookup(
        10.0, 75.0, expected_context=mixed.context
    ).status == DOMAIN_UNKNOWN_UNDER_RESOLVED
    assert coarse.lookup(
        10.0, 75.0, expected_context=coarse.context
    ).status == DOMAIN_UNKNOWN_UNDER_RESOLVED


def test_outside_and_context_mismatch_never_inherit_support():
    domain = _domain()

    assert domain.lookup(
        7.9, 75.0, expected_context=domain.context
    ).status == DOMAIN_UNKNOWN_OUTSIDE
    wrong_gamma = replace(domain.context, gamma=1.3)
    assert domain.lookup(
        10.0, 75.0, expected_context=wrong_gamma
    ).status == DOMAIN_CONTEXT_MISMATCH


def test_json_is_deterministic_hash_pinned_and_rejects_tampering(tmp_path):
    domain = _domain()
    first = domain.to_json()
    second = domain.to_json()
    path = tmp_path / "domain.json"
    domain.write_json(path)

    loaded = RaoConvergenceDomain.read_json(path)

    assert first == second
    assert "NaN" not in first
    assert loaded.artifact_sha256 == domain.artifact_sha256
    assert loaded.context.kernel_dtheta_limit_deg == 0.25
    assert loaded.context.kernel_massflow_rel_tol == 0.02
    payload = json.loads(first)
    payload["samples"][0][0]["reason"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        RaoConvergenceDomain.read_json(path)


def test_domain_rejects_unsorted_or_internally_inconsistent_evidence():
    domain = _domain()
    with pytest.raises(ValueError, match="strictly increasing"):
        replace(domain, epsilon_axis=(12.0, 8.0))
    with pytest.raises(ValueError, match="evaluable, valid Rao Eq"):
        replace(domain.samples[0][0], rao_eq5_valid=False)
    with pytest.raises(ValueError, match="validity source"):
        replace(domain, validity_source_id="not-the-canonical-eq5-source")


def test_kernel_march_controls_are_validated_and_part_of_context_identity():
    config = ExistenceScanConfig(Rt=0.02, epsilon=10.0)
    context = RaoConvergenceContext.from_scan_config(
        config, closure_model=MODEL_SMOOTH
    )

    assert context.kernel_dtheta_limit_deg == 0.25
    assert context.kernel_massflow_rel_tol == 0.02
    assert context.to_dict()["kernel_dtheta_limit_deg"] == 0.25
    assert context.to_dict()["kernel_massflow_rel_tol"] == 0.02
    assert replace(
        context, kernel_dtheta_limit_deg=0.125
    ).identity_sha256 != context.identity_sha256
    assert replace(
        context, kernel_massflow_rel_tol=0.01
    ).identity_sha256 != context.identity_sha256

    with pytest.raises(ValueError, match="kernel_dtheta_limit_deg"):
        replace(config, kernel_dtheta_limit_deg=0.0)
    with pytest.raises(ValueError, match="kernel_massflow_rel_tol"):
        replace(config, kernel_massflow_rel_tol=1.0001)


def test_large_finite_scan_residual_cannot_be_promoted_to_supported():
    domain = _domain()
    high_residual = replace(
        domain.samples[0][0], residual_norm_geometry=0.5
    )
    rows = (
        (high_residual, domain.samples[0][1]),
        domain.samples[1],
    )

    with pytest.raises(ValueError, match="exceeds the domain residual limit"):
        replace(domain, samples=rows)


def test_resolution_promotion_requires_every_kernel_rung_to_converge():
    with pytest.raises(ValueError, match="every rung"):
        replace(
            _resolution(),
            root_success=(False, True),
        )


def test_domain_rejects_sample_ladder_that_differs_from_context():
    domain = _domain()
    wrong_resolution = replace(
        domain.samples[0][0].resolution,
        n_kernel=(10, 12),
    )
    wrong_sample = replace(
        domain.samples[0][0], resolution=wrong_resolution
    )
    rows = (
        (wrong_sample, domain.samples[0][1]),
        domain.samples[1],
    )

    with pytest.raises(ValueError, match="ladder does not match"):
        replace(domain, samples=rows)


def test_builder_uses_external_axes_and_keeps_internal_roots_as_outputs(monkeypatch):
    sentinel_scan = object()
    monkeypatch.setattr(scan_module, "scan_existence", lambda config: sentinel_scan)
    monkeypatch.setattr(
        scan_module,
        "_scan_source_sha256",
        lambda result, model: "b" * 64,
    )
    monkeypatch.setattr(
        scan_module,
        "refine_from_scan_best",
        lambda result, model, maxfev: _root(12),
    )
    monkeypatch.setattr(
        scan_module,
        "resolution_convergence",
        lambda config, **kwargs: [_root(8, 25.0, 0.20), _root(12, 25.01, 0.2002)],
    )
    monkeypatch.setattr(
        scan_module,
        "evaluate_closure_point",
        lambda config, **kwargs: {
            "rao_eq5_evaluable": True,
            "rao_eq5_valid": True,
            "rao_eq5_min_dr_over_r": 0.01,
        },
    )
    config = ExistenceScanConfig(Rt=0.02, epsilon=10.0, length_pct=80.0)

    domain = scan_convergence_domain(
        config,
        epsilon_values=(8.0, 12.0),
        length_pct_values=(70.0, 80.0),
        n_kernel_values=(8, 12),
        max_epsilon_spacing=4.0,
        max_length_pct_spacing=10.0,
    )

    assert domain.epsilon_axis == (8.0, 12.0)
    assert domain.length_pct_axis == (70.0, 80.0)
    assert domain.samples[0][0].theta_b_deg == pytest.approx(25.01)
    assert domain.samples[0][0].kdf == pytest.approx(0.2002)
    assert domain.context.n_kernel_ladder == (8, 12)
    assert domain.context.kernel_dtheta_limit_deg == 0.25
    assert domain.context.kernel_massflow_rel_tol == 0.02
    assert domain.context.root_maxfev == 80
    assert domain.context.theta_b_seed_policy == "fixed_axis"
    assert domain.context.theta_b_seed_axis_deg == tuple(config.theta_axis)
    assert domain.context.kdf_seed_axis == tuple(config.kdf_axis)
    assert all(
        sample.classification == SAMPLE_CONVERGED_VALID
        for row in domain.samples
        for sample in row
    )


def test_builder_records_one_bad_scan_as_numerical_failure_and_continues(
    monkeypatch,
):
    sentinel_scan = object()

    def fake_scan(config):
        if config.epsilon == 8.0 and config.length_pct == 70.0:
            raise RuntimeError("synthetic cell failure")
        return sentinel_scan

    monkeypatch.setattr(scan_module, "scan_existence", fake_scan)
    monkeypatch.setattr(
        scan_module,
        "_scan_source_sha256",
        lambda result, model: "b" * 64,
    )
    monkeypatch.setattr(
        scan_module,
        "refine_from_scan_best",
        lambda result, model, maxfev: _root(12),
    )
    monkeypatch.setattr(
        scan_module,
        "resolution_convergence",
        lambda config, **kwargs: [_root(8), _root(12, 25.01, 0.2002)],
    )
    monkeypatch.setattr(
        scan_module,
        "evaluate_closure_point",
        lambda config, **kwargs: {
            "rao_eq5_evaluable": True,
            "rao_eq5_valid": True,
            "rao_eq5_min_dr_over_r": 0.01,
        },
    )
    config = ExistenceScanConfig(Rt=0.02, epsilon=10.0, length_pct=80.0)

    domain = scan_convergence_domain(
        config,
        epsilon_values=(8.0, 12.0),
        length_pct_values=(70.0, 80.0),
        n_kernel_values=(8, 12),
        max_epsilon_spacing=4.0,
        max_length_pct_spacing=10.0,
    )

    failed = domain.samples[0][0]
    assert failed.classification == SAMPLE_NUMERICAL_FAILURE
    assert "not evidence of physical non-existence" in failed.reason
    assert domain.samples[0][1].classification == SAMPLE_CONVERGED_VALID


def test_chart_centered_seed_policy_is_applied_per_external_knot(monkeypatch):
    from raosim.nozzle_geometry import lookup_angles

    sentinel_scan = object()
    seen_axes = {}

    def fake_scan(config):
        seen_axes[(config.epsilon, config.length_pct)] = tuple(
            float(value) for value in config.theta_axis
        )
        return sentinel_scan

    monkeypatch.setattr(scan_module, "scan_existence", fake_scan)
    monkeypatch.setattr(
        scan_module,
        "_scan_source_sha256",
        lambda result, model: "b" * 64,
    )
    monkeypatch.setattr(
        scan_module,
        "refine_from_scan_best",
        lambda result, model, maxfev: _root(12),
    )
    monkeypatch.setattr(
        scan_module,
        "resolution_convergence",
        lambda config, **kwargs: [_root(8), _root(12, 25.01, 0.2002)],
    )
    monkeypatch.setattr(
        scan_module,
        "evaluate_closure_point",
        lambda config, **kwargs: {
            "rao_eq5_evaluable": True,
            "rao_eq5_valid": True,
            "rao_eq5_min_dr_over_r": 0.01,
        },
    )
    config = ExistenceScanConfig(Rt=0.02, epsilon=10.0, length_pct=80.0)

    domain = scan_convergence_domain(
        config,
        epsilon_values=(8.0, 12.0),
        length_pct_values=(70.0, 80.0),
        n_kernel_values=(8, 12),
        max_epsilon_spacing=4.0,
        max_length_pct_spacing=10.0,
        theta_b_seed_policy="rao_chart_centered",
        theta_b_chart_span_deg=40.0,
        theta_b_chart_count=21,
    )

    for (epsilon, length_pct), axis in seen_axes.items():
        chart_theta_n, _ = lookup_angles(epsilon, length_pct)
        assert len(axis) == 21
        assert axis[0] == pytest.approx(chart_theta_n - 20.0)
        assert axis[-1] == pytest.approx(chart_theta_n + 20.0)
    assert domain.context.theta_b_seed_policy == "rao_chart_centered"
    assert domain.context.theta_b_seed_axis_deg == ()
    assert domain.context.theta_b_chart_span_deg == 40.0
    assert domain.context.theta_b_chart_count == 21


def test_rt_scale_invariance_of_root_and_declared_context():
    base = ExistenceScanConfig(
        Rt=0.02,
        epsilon=10.0,
        length_pct=80.0,
        n_kernel=8,
        n_de_points=8,
        models=(MODEL_SMOOTH,),
    )
    scaled = replace(base, Rt=0.04)

    root_a = refine_geometry_root(
        base,
        model=MODEL_SMOOTH,
        theta_B_seed_deg=25.5,
        kdf_seed=0.15,
        maxfev=50,
    )
    root_b = refine_geometry_root(
        scaled,
        model=MODEL_SMOOTH,
        theta_B_seed_deg=25.5,
        kdf_seed=0.15,
        maxfev=50,
    )
    eval_a = evaluate_closure_point(
        base,
        model=MODEL_SMOOTH,
        theta_B_deg=root_a.theta_B_deg,
        kdf=root_a.kdf,
    )
    eval_b = evaluate_closure_point(
        scaled,
        model=MODEL_SMOOTH,
        theta_B_deg=root_b.theta_B_deg,
        kdf=root_b.kdf,
    )

    assert root_a.success and root_b.success
    assert root_a.theta_B_deg == pytest.approx(root_b.theta_B_deg, abs=1.0e-10)
    assert root_a.kdf == pytest.approx(root_b.kdf, abs=1.0e-10)
    assert eval_a["rao_eq5_min_dr_over_r"] == pytest.approx(
        eval_b["rao_eq5_min_dr_over_r"], abs=1.0e-12
    )
    context_a = RaoConvergenceContext.from_scan_config(
        base, closure_model=MODEL_SMOOTH
    )
    context_b = RaoConvergenceContext.from_scan_config(
        scaled, closure_model=MODEL_SMOOTH
    )
    assert context_a.Rt_reference != context_b.Rt_reference
    assert context_a.identity_sha256 == context_b.identity_sha256


def test_tracked_v1_fixed_slice_artifact_fails_closed_until_regenerated():
    repo = Path(__file__).resolve().parent.parent
    path = (
        repo
        / "raosim"
        / "data"
        / "rao_domains"
        / "rao_d_attachment_fixed_slice_gamma1p4_pa0p01_eps_length_v1.json"
    )

    with pytest.raises(ValueError, match="legacy convergence-domain schema v1"):
        RaoConvergenceDomain.read_json(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert "kernel_dtheta_limit_deg" not in payload["context"]
    assert "kernel_massflow_rel_tol" not in payload["context"]


def test_tracked_v2_fixed_slice_artifact_has_pinned_complete_identity():
    repo = Path(__file__).resolve().parent.parent
    path = (
        repo
        / "raosim"
        / "data"
        / "rao_domains"
        / "rao_d_attachment_fixed_slice_gamma1p4_pa0p01_eps_length_v2.json"
    )

    domain = RaoConvergenceDomain.read_json(path)
    assert domain.artifact_sha256 == (
        "2b2d1f4017f88d230cd6857dbc1e1a859ac5d3f611aa14a60f9e413d2c5949a9"
    )
    assert domain.context.identity_sha256 == (
        "a3095bad66a56a24554f7e234b9133f967110ee4958998193d385f32d5d7d277"
    )
    assert domain.context.kernel_dtheta_limit_deg == 0.25
    assert domain.context.kernel_massflow_rel_tol == 0.02
    assert domain.classification_counts() == {
        "converged_valid": 40,
        "converged_invalid": 0,
        "not_converged": 25,
        "numerical_failure": 0,
        "unevaluable": 0,
        "under_resolved": 0,
    }

    midpoint_statuses = Counter(
        domain.lookup(
            0.5 * (epsilon_left + epsilon_right),
            0.5 * (length_left + length_right),
            expected_context=domain.context,
        ).status
        for epsilon_left, epsilon_right in zip(
            domain.epsilon_axis[:-1], domain.epsilon_axis[1:]
        )
        for length_left, length_right in zip(
            domain.length_pct_axis[:-1], domain.length_pct_axis[1:]
        )
    )
    assert midpoint_statuses == {
        "supported": 23,
        "refused_not_converged": 11,
        "unknown_under_resolved": 14,
    }


def test_cli_writes_hash_valid_domain_and_rejects_unpaired_axes(tmp_path):
    repo = Path(__file__).resolve().parent.parent
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    output_dir = tmp_path / "domain-output"
    common = [
        sys.executable,
        "scripts/rao_existence_scan.py",
        "--theta-b-values-deg",
        "25.0",
        "--kdf-values",
        "0.15",
        "--models",
        "smooth",
        "--n-kernel",
        "5",
        "--kernel-dtheta-limit-deg",
        "0.375",
        "--kernel-massflow-rel-tol",
        "0.015",
        "--n-de-points",
        "5",
        "--root-maxfev",
        "15",
        "--resolution-convergence",
        "5,6",
        "--domain-epsilon-values",
        "9,10",
        "--domain-length-pct-values",
        "75,80",
        "--domain-max-epsilon-spacing",
        "1",
        "--domain-max-length-pct-spacing",
        "5",
        "--output-dir",
        str(output_dir),
    ]

    completed = subprocess.run(
        common,
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "Domain sample classifications:" in completed.stdout
    domain_path = output_dir / "convergence_domain.json"
    loaded = RaoConvergenceDomain.read_json(domain_path)
    assert loaded.to_dict()["artifact_sha256"] == loaded.artifact_sha256
    assert loaded.context.kernel_dtheta_limit_deg == 0.375
    assert loaded.context.kernel_massflow_rel_tol == 0.015

    incomplete = subprocess.run(
        [
            sys.executable,
            "scripts/rao_existence_scan.py",
            "--domain-epsilon-values",
            "9,10",
            "--output-dir",
            str(tmp_path / "incomplete"),
        ],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert incomplete.returncode != 0
    assert "requires both --domain-epsilon-values" in incomplete.stderr
