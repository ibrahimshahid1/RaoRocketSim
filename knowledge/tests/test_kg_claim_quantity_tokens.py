"""English connectors must not create physical-quantity associations."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kg import harvest_extraction


def test_connector_of_does_not_become_mixture_ratio(monkeypatch):
    seen = []

    def resolve(token):
        seen.append(token)
        return ({"state": "RESOLVED", "candidates": [{"qid": "QTY_MR"}]}
                if token.lower() in {"of", "mr"} else {"state": "UNRESOLVED_ENTITY"})

    monkeypatch.setattr(harvest_extraction, "resolve_symbol", resolve)
    assert harvest_extraction.claim_quantity_ids(
        {"subject": "bell nozzle", "object": "80 percent of equivalent cone"}) == []
    assert "of" not in seen
    assert harvest_extraction.claim_quantity_ids(
        {"subject": "MR", "object": "mixture ratio"}) == ["QTY_MR"]
