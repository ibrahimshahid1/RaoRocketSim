"""Replay the completed 2026-10-16 audit journals and verify preserved content.

Run with the repository's scientific Python environment. This deliberately banks
the twelve named journals and rebuilds derived indexes. It checks the saved White
relation inventory independently of replay stability: stable missing data must
not pass. This is a bounded audit fixture, not a corpus completeness certificate.
"""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "knowledge"))
from kg.coverage import summary
from kg.ground import normalize
from kg.store import load

JOURNALS = (
    "lee-docling-2026-10-15.jsonl", "nasa8089-docling-2026-10-15.jsonl",
    "white-boundary-docling-2026-10-15.jsonl", "white-integrals-docling-2026-10-15.jsonl",
    "white-gas-docling-2026-10-15.jsonl", "white-meter-docling-2026-10-15.jsonl",
    "white-area-docling-2026-10-16.jsonl", "white-choke-docling-2026-10-16.jsonl",
    "white-model-comparisons-2026-10-16.jsonl", "white-derivation-links-2026-10-16.jsonl",
    "anderson-frozen-docling-2026-10-16.jsonl", "anderson-frozen-derivations-2026-10-16.jsonl",
)
NAMES = ("manifest", "equations", "claims", "entities", "validity", "models",
         "software_links", "quantities", "provenance", "enrichments", "relationships",
         "artifacts", "requirements", "tests", "capabilities", "repo_documents")
FIELDS = ("id", "qid", "type", "form", "statement", "prose", "value", "units", "name",
          "source_sha256", "source_state", "active", "verification", "verifier_source_sha256",
          "model_ids", "model_id", "equation_ids", "assumption_ids", "variables", "machine_readable",
          "basis_class", "file", "line", "end_line", "qualname", "definition_sha256",
          "source_support_state", "source_support_sha256", "source_comparison_equation_ids",
          "review_state", "raw_sha256", "metadata_sha256", "metadata_integrity_state", "attempt_id")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def snapshot():
    result, duplicates = {}, {}
    for name in NAMES:
        rows = load(name)
        ids = [row.get("id") or row.get("qid") for row in rows]
        assert all(ids), name
        if len(ids) != len(set(ids)):
            duplicates[name] = len(ids) - len(set(ids))
        projection = [{key: row[key] for key in FIELDS if key in row} for row in rows]
        result[name] = digest(sorted(projection, key=lambda row: row.get("id") or row["qid"]))
    # Relationship timestamps, notes and retained history must also survive replay.
    result["relationship_complete_records"] = digest(sorted(load("relationships"), key=lambda row: row["id"]))
    graph = json.loads((ROOT / "knowledge/graph/graph.json").read_text())
    result["graph_node_identities"] = digest(sorted((row["id"], row["type"]) for row in graph["nodes"]))
    result["graph_edges"] = digest(sorted(graph["edges"], key=lambda row: (row["from"], row["to"], row["type"])))
    return result, duplicates


def check_derivation_inventory(journal_name="white-derivation-links-2026-10-16.jsonl",
                               result_key="white-explicit-derivations", expected_count=34):
    journal = ROOT / "knowledge/workflows/results" / journal_name
    effective = {}
    for line in journal.read_text().splitlines():
        row = json.loads(line)
        if row.get("type") == "result":
            effective[row["key"]] = row["value"]
    relations = effective[result_key]["cross_document_relations"]
    equations = load("equations")
    docs = {row["id"]: row for row in load("manifest")}
    by_label = {}
    for equation in equations:
        by_label.setdefault((equation["document_id"], normalize(equation.get("label", ""))), []).append(equation)
    expected = []
    endpoints = set()
    for relation in relations:
        ids = []
        for side in ("a", "b"):
            matches = by_label.get((relation[side + "_doc_id"], normalize(relation[side + "_label"])), [])
            assert len(matches) == 1, (side, relation, "ambiguous or missing source equation")
            equation = matches[0]
            doc = docs[equation["document_id"]]
            assert equation.get("active", True) and equation.get("source_state") == "CURRENT", equation["id"]
            assert equation.get("source_sha256") == doc["sha256"], equation["id"]
            assert equation.get("verification") == "CONFIRMED", equation["id"]
            assert equation.get("verifier_source_sha256") == doc["sha256"], equation["id"]
            ids.append(equation["id"])
            endpoints.add(equation["id"])
        expected.append((ids[0], relation["relation"], ids[1]))
    assert len(expected) == len(set(expected)) == expected_count, "The saved relation inventory changed"
    stored = load("relationships")
    stored_counts = Counter((row["from"], row["edge"], row["to"]) for row in stored)
    graph = json.loads((ROOT / "knowledge/graph/graph.json").read_text())
    graph_counts = Counter((row["from"], row["type"], row["to"]) for row in graph["edges"])
    nodes = {row["id"] for row in graph["nodes"]}
    assert endpoints <= nodes, "A source endpoint is missing from the graph"
    for identity in expected:
        assert stored_counts[identity] == 1, (identity, "missing or duplicated stored relationship")
        assert graph_counts[identity] == 1, (identity, "missing or duplicated graph relationship")
        row, = [row for row in stored if (row["from"], row["edge"], row["to"]) == identity]
        rid = "REL_" + hashlib.sha256(json.dumps(list(identity), ensure_ascii=False,
                                                separators=(",", ":")).encode()).hexdigest()
        assert row["id"] == rid, (identity, "relationship ID does not bind the complete tuple")
    return {"status": "PASS", "journal_sha256": hashlib.sha256(journal.read_bytes()).hexdigest(),
            "expected_unique_tuples": len(expected), "stored_exactly_once": len(expected),
            "graph_exactly_once": len(expected), "current_confirmed_endpoints": len(endpoints),
            "by_type": dict(Counter(identity[1] for identity in expected)),
            "scope": "Saved explicit same-source comparisons; no cross-source completeness assertion."}


def main():
    os.chdir(ROOT)
    before, duplicates = snapshot()
    assert not duplicates, duplicates
    cardinality_before = check_derivation_inventory()
    anderson_before = check_derivation_inventory("anderson-frozen-derivations-2026-10-16.jsonl",
                                                "anderson-explicit-derivations", 5)
    replays = []
    for name in JOURNALS:
        cmd = [sys.executable, "knowledge/tools/bank_extraction.py", "knowledge/workflows/results/" + name]
        if "derivation" in name:
            cmd.append("--grouped")
        result = subprocess.run(cmd, text=True, capture_output=True, check=True)
        replays.append({"journal": name, "exit_code": result.returncode,
                        "journal_sha256": hashlib.sha256((ROOT / "knowledge/workflows/results" / name).read_bytes()).hexdigest(),
                        "state": "ALREADY_BANKED_NO_CHANGES" if "All results already banked; no changes" in result.stdout else "REAPPLIED",
                        "output": result.stdout.strip()})
    after_bank, duplicates = snapshot()
    assert after_bank == before and not duplicates, "Source replay changed audited content"
    result = subprocess.run(["knowledge/kg.sh", "rebuild"], text=True, capture_output=True, check=True)
    (ROOT / "knowledge/.work/replay_rebuild_2026-10-16.json").write_text(result.stdout)
    after, duplicates = snapshot()
    changed = [key for key in before if before[key] != after[key]]
    cardinality_after = check_derivation_inventory()
    anderson_after = check_derivation_inventory("anderson-frozen-derivations-2026-10-16.jsonl",
                                               "anderson-explicit-derivations", 5)
    report = {"date": "2026-10-16", "recorded_at": datetime.now(timezone.utc).isoformat(),
              "status": "FAIL" if changed or duplicates else "PASS",
              "scope": "Current rebuild plus replay of twelve completed remediation journals. Projections compare identities, source/active/verification/support states, math forms, variable bindings, locators, defaults and complete graph edges. Complete relationship records and all 34 White plus 5 Anderson saved relation tuples are checked separately. Generated store metadata timestamps are excluded. This is software reproducibility, not scientific completeness.",
              "duplicate_id_groups": duplicates, "changed_projections": changed,
              "before": before, "after": after, "journal_replays": replays,
              "derivation_inventory_before": cardinality_before,
              "derivation_inventory_after": cardinality_after,
              "anderson_derivation_inventory_before": anderson_before,
              "anderson_derivation_inventory_after": anderson_after, "coverage": summary()}
    (ROOT / "knowledge/reports/audit_replay_validation_2026-10-16.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"status": report["status"], "changed_projections": changed,
                      "duplicate_id_groups": duplicates, "journals": len(replays),
                      "derivation_inventory": cardinality_after,
                      "anderson_derivation_inventory": anderson_after}))
    return 1 if changed or duplicates else 0


if __name__ == "__main__":
    raise SystemExit(main())
