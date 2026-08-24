"""Two scope gates: separation topology, and milling practice.

Neither adds physics. Both convert a silent assumption into a stated one, which
is the cheaper half of the fidelity roadmap and the half that stops a result
being read as more than it is.
"""

from __future__ import annotations

import pytest

from raosim.manufacturing_limits import (
    MILLED_CHANNEL_LIMITS,
    check_channel_geometry,
    check_design_space,
)
from raosim.mdo.metrics import MetricNotAvailable, resolve_metric


# --------------------------------------------------------------------------- #
# Separation: the correlations are free-shock-only                             #
# --------------------------------------------------------------------------- #
def test_separation_result_declares_its_topology_scope():
    """An unqualified "not separated" must not read as a full verdict.

    Frey & Hagemann (JPP 16(3), 2000) report restricted shock separation in
    full-scale thrust-optimized nozzles -- the contour family this repository
    generates -- and an onset pressure ratio cannot represent it.
    """
    import numpy as np

    from raosim.separation import SEPARATION_PATTERN_SCOPE, check_separation

    contour = {
        "x": np.linspace(0.0, 0.3, 60),
        "y": np.linspace(0.02, 0.08, 60),
        "Rt": 0.02,
        "epsilon": 16.0,
    }
    result = check_separation(contour, Pc=3.0e6, Pa=101325.0, gamma=1.24)

    assert result["separation_pattern_scope"] == SEPARATION_PATTERN_SCOPE
    assert result["separation_pattern_scope"] == "free_shock_separation_only"
    assert result["restricted_shock_separation"] == "not_modelled"
    assert "Frey" in result["restricted_shock_separation_reason"]
    assert result["scope_source_id"].startswith("Frey-Hagemann-2000")


def test_side_load_is_reported_unavailable_not_estimated():
    """A correlation would not be honest: the driver is the FSS/RSS transition."""
    import numpy as np

    from raosim.separation import check_separation

    contour = {
        "x": np.linspace(0.0, 0.3, 60),
        "y": np.linspace(0.02, 0.08, 60),
        "Rt": 0.02,
        "epsilon": 16.0,
    }
    result = check_separation(contour, Pc=3.0e6, Pa=101325.0, gamma=1.24)

    assert result["side_load_n"] is None
    assert result["side_load_unavailable_reason"]


def test_side_load_metric_is_registered_and_blocked():
    with pytest.raises(MetricNotAvailable, match="FSS"):
        resolve_metric("aerodynamics.side_load", role="target")
    with pytest.raises(MetricNotAvailable):
        resolve_metric("aerodynamics.side_load", role="objective")


def test_aspect_ratio_constraint_is_sourced_and_labelled_manufacturing():
    """AR<=8 is a milling bound, not a model-validity bound."""
    from raosim.mdo.constraints import CONSTRAINT_MANIFEST

    spec = next(r for r in CONSTRAINT_MANIFEST if r.name == "aspect_ratio")
    assert spec.category == "manufacturing"
    assert spec.source_id is not None
    assert "Wadel-1997" in spec.source_id
    assert "Carlile-1992" in spec.source_id


# --------------------------------------------------------------------------- #
# Manufacturing limits (Wadel 1997 milled-channel constraints)                 #
# --------------------------------------------------------------------------- #
def test_limits_match_the_published_milled_channel_list():
    by_name = {limit.quantity: limit for limit in MILLED_CHANNEL_LIMITS}
    # "widths >0.051 cm (0.02 in.)"
    assert by_name["channel_width"].minimum == pytest.approx(0.51e-3)
    # "heights <0.51 cm (0.20 in.)"
    assert by_name["channel_height"].maximum == pytest.approx(5.1e-3)
    # "landwidths >0.051 cm (0.02 in.)"
    assert by_name["land_width"].minimum == pytest.approx(0.51e-3)
    # "Aspect ratios <8"
    assert by_name["channel_aspect_ratio"].maximum == pytest.approx(8.0)
    for limit in MILLED_CHANNEL_LIMITS:
        assert "Wadel-1997" in limit.source_id
        assert "extrapolation" in limit.applicability


def test_the_default_search_box_admits_unmillable_channel_width():
    """The finding this gate exists to surface.

    The design space allows a 0.30 mm channel, below the 0.51 mm milled
    minimum. That is not a physics violation, so it is reported rather than
    enforced -- but a box corner that cannot be milled will be explored.
    """
    from raosim.mdo.schema import default_design_space

    findings = check_design_space(default_design_space())
    widths = [f for f in findings if f.quantity == "channel_width"]
    assert widths, "expected the channel-width lower bound to be flagged"
    assert widths[0].kind == "search_box"
    assert "cannot be milled" in widths[0].message


def test_a_resolved_section_is_checked_against_every_limit():
    findings = check_channel_geometry(
        channel_width=0.30e-3, channel_height=3.0e-3, land_width=0.20e-3,
    )
    flagged = {f.quantity for f in findings}
    assert flagged == {"channel_width", "land_width", "channel_aspect_ratio"}


def test_aspect_ratio_is_derived_not_taken_on_trust():
    """A caller cannot claim a ratio that its own width and height contradict."""
    findings = check_channel_geometry(channel_width=0.6e-3, channel_height=6.0e-3)
    ratios = [f for f in findings if f.quantity == "channel_aspect_ratio"]
    assert ratios and "exceeds" in ratios[0].message


def test_a_millable_section_produces_no_findings():
    assert check_channel_geometry(
        channel_width=0.6e-3, channel_height=3.0e-3, land_width=0.6e-3,
    ) == ()


def test_unknown_process_is_rejected_rather_than_assumed_millable():
    with pytest.raises(ValueError, match="no manufacturing limits"):
        check_channel_geometry(channel_width=1.0e-3, process="cast")


def test_job_resolution_surfaces_manufacturability_in_its_notes():
    from raosim.mdo.job import (
        ArchitectureSelection,
        EngineJobSpec,
        MissionRequest,
        ObjectiveRequest,
        resolve_engine_job,
    )

    job = resolve_engine_job(EngineJobSpec(
        mission=MissionRequest(thrust="13 kN"),
        workflow="optimize",
        objective=ObjectiveRequest("masses.dry_mass_partial", "min"),
        architecture=ArchitectureSelection(liner_material="GRCop-84"),
    ))
    notes = [n for n in job.notes if n.startswith("manufacturability:")]
    assert notes, "search-box manufacturability must be reported"
    assert any("Wadel-1997" in n for n in notes)
