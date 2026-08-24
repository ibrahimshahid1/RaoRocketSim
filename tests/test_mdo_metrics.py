"""The selectable-output registry, and its two structural rules.

Constraint metrics are *derived* from the constraint manifest rather than
re-listed, and every reportable metric is validated against the snapshot
contract. Both rules exist so the registry cannot become a third competing
definition of quantities the repository already names elsewhere.
"""

from __future__ import annotations

import pytest

from raosim.mdo.constraints import CONSTRAINT_MANIFEST
from raosim.mdo.metrics import (
    METRIC_REGISTRY,
    MetricNotAvailable,
    objective_capable_metrics,
    resolve_metric,
    target_capable_metrics,
    unavailable_metrics,
)
from raosim.mdo.snapshot import SNAPSHOT_FIELD_MANIFEST


def test_every_snapshot_path_in_the_registry_actually_exists():
    """A metric pointing at a nonexistent field is worse than no metric."""
    for metric in METRIC_REGISTRY.values():
        if metric.snapshot_section is None:
            continue
        assert metric.snapshot_section in SNAPSHOT_FIELD_MANIFEST, metric.id
        assert (
            metric.snapshot_field
            in SNAPSHOT_FIELD_MANIFEST[metric.snapshot_section]
        ), metric.id


def test_constraint_metrics_are_derived_from_the_single_manifest():
    """One row per manifest constraint, no more and no fewer."""
    derived = {
        m.id: m for m in METRIC_REGISTRY.values()
        if m.category.startswith("constraint:")
    }
    assert len(derived) == len(CONSTRAINT_MANIFEST)
    for spec in CONSTRAINT_MANIFEST:
        metric = derived[f"constraints.{spec.name}_margin"]
        assert metric.engine_key == spec.engine_key
        assert metric.source_id == spec.source_id


def test_constraints_cannot_be_retargeted():
    """Re-targeting an enforced limit would silently weaken it."""
    for metric in METRIC_REGISTRY.values():
        if metric.category.startswith("constraint:"):
            assert not metric.target_capable, metric.id
            assert not metric.objective_capable, metric.id
    with pytest.raises(MetricNotAvailable, match="must not be re-targeted"):
        resolve_metric("constraints.wall_temp_margin", role="target")


def test_objectives_are_differentiable_available_and_honestly_named():
    from raosim.mdo.execute import _OBJECTIVE_IMPLEMENTATIONS

    objectives = {m.id for m in objective_capable_metrics()}
    assert objectives == {
        "masses.dry_mass_partial",
        "masses.electric_package",
    }
    assert objectives == set(_OBJECTIVE_IMPLEMENTATIONS)
    # Nothing may claim to be a whole-engine mass objective.
    assert "masses.engine_dry_mass" not in objectives
    for metric in objective_capable_metrics():
        assert metric.differentiable and metric.available, metric.id


def test_engineering_metrics_carry_their_physical_dimensions():
    assert METRIC_REGISTRY["geometry.chamber_volume"].dimension == "volume"
    assert METRIC_REGISTRY["thermal.heat_flux_max"].dimension == "heat_flux"


def test_unmodellable_quantities_are_registered_with_a_reason_not_omitted():
    """Omission reads as 'not considered'; a reason can be explained."""
    unavailable = {m.id: m for m in unavailable_metrics()}
    for expected in (
        "masses.engine_dry_mass",
        "performance.engine_system_isp",
        "life.qualified_reusable_cycles",
        "life.cycles_to_failure",
        "stability.combustion",
        "cost.recurring",
        "reliability.probability",
    ):
        assert expected in unavailable
        assert unavailable[expected].unavailable_reason
    with pytest.raises(MetricNotAvailable, match="valves, lines"):
        resolve_metric("masses.engine_dry_mass", role="objective")


def test_cycles_to_failure_is_explicitly_unavailable_for_both_roles():
    metric = METRIC_REGISTRY["life.cycles_to_failure"]
    assert not metric.available
    assert "static screening stress" in (metric.unavailable_reason or "")
    assert "duty cycle" in (metric.unavailable_reason or "")
    with pytest.raises(MetricNotAvailable, match="not a cycle count"):
        resolve_metric("life.cycles_to_failure", role="target")
    with pytest.raises(MetricNotAvailable, match="not a cycle count"):
        resolve_metric("life.cycles_to_failure", role="objective")


def test_unknown_metrics_are_rejected_with_a_pointer_to_the_catalog():
    with pytest.raises(LookupError, match="unknown metric"):
        resolve_metric("performance.warp_factor")


def test_target_capable_set_is_non_trivial_and_available():
    targets = target_capable_metrics()
    assert len(targets) >= 10
    for metric in targets:
        assert metric.available, metric.id


def test_registry_ids_are_unique_and_namespaced():
    for metric_id in METRIC_REGISTRY:
        assert "." in metric_id, metric_id
    assert len(set(METRIC_REGISTRY)) == len(METRIC_REGISTRY)
