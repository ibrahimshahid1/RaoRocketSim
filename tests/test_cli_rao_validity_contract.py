from dataclasses import replace
import math
from types import SimpleNamespace

import pytest

from raosim.moc import FlowNode
from raosim.rao_variational import rao_radial_increment_validity
from raosim.run_nozzle import _rao_radial_validity_gate


@pytest.mark.parametrize(("states", "expected"), [
    ([(2.0, 60.0, 0.025), (4.0, 55.0, 0.027)], True),
    ([(4.0, 55.0, 0.027), (5.0, 40.0, 0.030)], False),
])
def test_cli_eq5_gate_uses_typed_verdict_when_legacy_boundary_disagrees(states, expected):
    nodes = [
        FlowNode(x=index * 0.005, r=radius, M=mach, theta=math.radians(theta))
        for index, (mach, theta, radius) in enumerate(states)
    ]
    validity = rao_radial_increment_validity(nodes)
    assert validity.available
    assert validity.valid is expected
    legacy_boundary = min(validity.boundary_function_eq6)
    assert (legacy_boundary >= -0.002) is not expected
    solution = SimpleNamespace(
        rao_validity=validity,
        construction_diagnostics={"boundary_min": legacy_boundary},
    )
    passed, payload = _rao_radial_validity_gate(solution, bvp_mode=True)
    assert passed is expected
    assert payload["valid"] is expected


def test_cli_missing_typed_eq5_evidence_refuses_even_if_legacy_diagnostic_passes():
    solution = SimpleNamespace(
        rao_validity=None,
        construction_diagnostics={
            "boundary_min": 1.0,
            "rao_radial_increment_validity": {"available": True, "valid": True},
        },
    )
    passed, payload = _rao_radial_validity_gate(solution, bvp_mode=True)
    assert not passed
    assert payload["available"] is False
    assert payload["status"] == "missing_typed_eq5_validity"


def test_cli_unavailable_eq5_refuses_and_top_path_is_not_applicable():
    validity = rao_radial_increment_validity([])
    solution = SimpleNamespace(rao_validity=replace(validity, valid=True))
    passed, payload = _rao_radial_validity_gate(solution, bvp_mode=True)
    assert not passed
    assert payload["available"] is False
    assert _rao_radial_validity_gate(SimpleNamespace(), bvp_mode=False) == (True, None)
