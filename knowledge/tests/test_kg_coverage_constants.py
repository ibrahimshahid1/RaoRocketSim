"""Coverage gaps describe current constants while preserving stored history."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kg import coverage


def constant(identity, **fields):
    return {"id": identity, "type": "EmpiricalConstant", **fields}


def test_historical_unknown_basis_does_not_inflate_current_gap():
    result = coverage.summary({"software_links": [
        constant("CONST_CURRENT", basis_class="UNKNOWN_BASIS", active=True, source_state="CURRENT"),
        constant("CONST_RETIRED", basis_class="UNKNOWN_BASIS", active=False, source_state="CURRENT"),
        constant("CONST_STALE", basis_class="UNKNOWN_BASIS", active=True, source_state="STALE_SOURCE"),
        constant("CONST_BOTH", basis_class="UNKNOWN_BASIS", active=False, source_state="REMOVED"),
        constant("CONST_SUPPORTED", basis_class="LITERATURE_DERIVED", source_state="CURRENT"),
        constant("CONST_OLD_SUPPORTED", basis_class="LITERATURE_DERIVED", source_state="STALE_SOURCE"),
        {"id": "IMPL_CURRENT", "type": "SoftwareComponent", "basis_class": "UNKNOWN_BASIS"},
    ]})
    stats = result["software_constants"]
    assert stats["records"] == 6
    assert stats["by_basis"] == {"UNKNOWN_BASIS": 4, "LITERATURE_DERIVED": 2}
    assert stats["unknown_basis"] == 4  # stored total remains backward compatible
    assert stats["current"] == 2
    assert stats["inactive_or_noncurrent"] == 4
    assert stats["by_basis_current"] == {"UNKNOWN_BASIS": 1, "LITERATURE_DERIVED": 1}
    assert stats["by_basis_inactive_or_noncurrent"] == {"UNKNOWN_BASIS": 3, "LITERATURE_DERIVED": 1}
    assert stats["unknown_basis_current"] == result["gaps"]["unknown_basis_software_constants"] == 1
    assert stats["unknown_basis_inactive_or_noncurrent"] == 3


def test_legacy_state_defaults_and_unspecified_basis_stay_explicit():
    result = coverage.summary({"software_links": [
        constant("CONST_LEGACY"),
        constant("CONST_NULL_BASIS", basis_class=None),
        constant("CONST_EMPTY_BASIS", basis_class=""),
        constant("CONST_UNRESOLVED_STATE", basis_class="UNKNOWN_BASIS", source_state="UNKNOWN"),
    ]})
    stats = result["software_constants"]
    assert stats["records"] == stats["unknown_basis"] == 4
    assert stats["current"] == stats["unknown_basis_current"] == 3
    assert stats["inactive_or_noncurrent"] == stats["unknown_basis_inactive_or_noncurrent"] == 1
    assert result["gaps"]["unknown_basis_software_constants"] == 3


def test_empty_snapshot_has_explicit_zero_counts_without_loading_stores(monkeypatch):
    def unexpected_load(*args, **kwargs):
        raise AssertionError("Synthetic coverage must not read the private corpus")
    monkeypatch.setattr(coverage.store, "load", unexpected_load)
    stats = coverage.summary({})["software_constants"]
    assert stats["records"] == stats["current"] == stats["inactive_or_noncurrent"] == 0
    assert stats["unknown_basis"] == stats["unknown_basis_current"] == stats["unknown_basis_inactive_or_noncurrent"] == 0
    assert stats["by_basis"] == stats["by_basis_current"] == stats["by_basis_inactive_or_noncurrent"] == {}
