from __future__ import annotations

import json

from scripts.compare_nasa_reference import build_arg_parser, compare, write_artifacts


def test_compare_report_declares_fixture_overlay_provenance(tmp_path):
    args = build_arg_parser().parse_args([
        "--skip-solve",
        "--kernel-n",
        "101",
    ])
    report = compare(args)
    diagnostics = report["diagnostics"]

    assert diagnostics["canonical_reference_track"] == "visible_source_port"
    assert diagnostics["comparison_track"] == "historical_fixture_overlay"
    assert diagnostics["source_port_matched"] is None
    assert diagnostics["source_port_match_status"] == "not_evaluated_by_historical_fixture_overlay"
    assert diagnostics["source_port_workflow_complete"] is False
    assert diagnostics["source_port_workflow_status"] == "incomplete"
    assert diagnostics["fixture_overlay_available"] is True
    assert diagnostics["fixture_overlay_is_promotion_authority"] is False
    assert diagnostics["fixture_generator_provenance"] == "unresolved"
    assert diagnostics["nasa_reference_matched_eligible"] is False
    assert diagnostics["python_kernel_complete"] is False
    assert diagnostics["python_source_contour_available"] is False
    assert diagnostics["python_source_contour_complete"] is False
    assert diagnostics["python_bfe_overlay_available"] is False
    assert diagnostics["python_bfe_overlay_complete"] is False
    assert diagnostics["python_bfe_overlay_reason"] == "kernel used fallback BD construction"
    assert report["python"]["kernel_status"] == "partial"
    assert report["python"]["kernel_complete"] is False
    assert report["python"]["source_contour_status"] == "unavailable"
    assert report["python"]["bfe_status"] == "unavailable"
    assert (
        "current Python kernel did not reach the throat-arc wall"
        in diagnostics["nasa_reference_matched_blockers"]
    )
    assert not any(
        "fixture_generator_provenance" in blocker
        for blocker in diagnostics["nasa_reference_matched_blockers"]
    )

    written = write_artifacts(report, tmp_path)
    assert tmp_path / "report.json" in written
    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["diagnostics"] == diagnostics


def test_compare_report_marks_sauer_kernel_complete_without_reference_promotion():
    args = build_arg_parser().parse_args([
        "--skip-solve",
        "--kernel-n",
        "101",
        "--starting-line-method",
        "sauer_modified",
    ])
    report = compare(args)
    diagnostics = report["diagnostics"]

    assert report["python"]["kernel_status"] == "ok"
    assert report["python"]["kernel_complete"] is True
    assert report["python"]["bfe_status"] == "ok"
    assert report["python"]["bfe_grid_rows"] > 0
    assert report["python"]["bfe_wall_points"] == report["python"]["bfe_grid_rows"]
    assert diagnostics["python_kernel_complete"] is True
    assert diagnostics["python_source_contour_available"] is True
    assert diagnostics["python_source_contour_complete"] is True
    assert diagnostics["python_source_contour"]["length_closed"] is False
    assert diagnostics["python_source_contour"]["crop_nozzle_to_length"] == "not_ported"
    assert diagnostics["python_bfe_overlay_available"] is True
    assert diagnostics["python_bfe_overlay_complete"] is True
    assert diagnostics["source_port_workflow_complete"] is True
    assert diagnostics["source_port_workflow_status"] == "complete_not_certified"
    assert diagnostics["fixture_overlay_is_promotion_authority"] is False
    assert diagnostics["nasa_reference_matched_eligible"] is False
    assert report["python"]["source_contour_status"] == "ok"
    assert report["python"]["source_contour_length_closed"] is False
    assert report["python"]["source_contour_wall_points"] > report["python"]["bfe_wall_points"]
    assert any(
        metric["name"] == "BFE_Kernel.out.x.station_rms"
        for metric in report["metrics"]
    )
    assert any(
        metric["name"] == "BFE_wall_contour.x.station_rms"
        for metric in report["metrics"]
    )
    assert (
        "current Python kernel used fallback BD construction"
        not in diagnostics["nasa_reference_matched_blockers"]
    )
    assert (
        "current Python kernel did not reach the throat-arc wall"
        not in diagnostics["nasa_reference_matched_blockers"]
    )
