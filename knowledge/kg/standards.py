"""Standards as typed sources: held documents or explicit BLOCKED_ON_STANDARD.

The registry lists standards that repository requests or the consolidated
review require tolerance, inspection, fatigue-test or uncertainty rules to
cite. A standard is ``HELD`` only when an exact manifest document with a
current fingerprint is mapped to it; otherwise it stays ``BLOCKED_ON_STANDARD``.
Editions are recorded only where the held document identifies one; requested
but unheld standards keep ``edition: UNSPECIFIED`` rather than a guessed year.
Holding a PDF is not compliance, and no rule-level link is inferred here.
"""

from __future__ import annotations

from datetime import datetime, timezone

from . import store

REGISTRY = (
    {"id": "STD_ASME_Y14_5", "designation": "ASME Y14.5", "title": "Dimensioning and Tolerancing",
     "held_document_id": "DOC_ASME_Y14_5_2018_DIMENSIONING_AND_TOLERANCING_8_2_THE_AMERICA", "held_edition": "2018",
     "requested_by": ["consolidated review section 10.4 item 8", "docs/LITERATURE_REQUESTS_HARDWARE_LOOP.md H6/GD&T note"],
     "capability_ids": []},
    {"id": "STD_ASME_Y14_41", "designation": "ASME Y14.41", "title": "Digital Product Definition Data Practices",
     "held_document_id": "DOC_ASME_Y14_41_2003_DIGITAL_PRODUCT_DEFINITION_DATA_PRACTICES_T", "held_edition": "2003",
     "requested_by": ["consolidated review section 10.4 item 8"], "capability_ids": []},
    {"id": "STD_ASME_Y14_24", "designation": "ASME Y14.24", "title": "Types and Applications of Engineering Drawings",
     "held_document_id": "DOC_ASME_Y14_24_2012_TYPES_AND_APPLICATIONS_OF_ENGINEERING_THE_A", "held_edition": "2012",
     "requested_by": ["docs/LITERATURE_REQUESTS_HARDWARE_LOOP.md GD&T note"], "capability_ids": []},
    {"id": "STD_ISO_286", "designation": "ISO 286", "title": "ISO system of limits and fits",
     "held_document_id": None, "requested_by": ["consolidated review section 10.4 item 8",
                                                 "docs/LITERATURE_REQUESTS_HARDWARE_LOOP.md H6"],
     "capability_ids": ["CAP_LITREQ_DOCS_LITERATURE_REQUESTS_HARDWARE_LOOP_MD_H6_ACHIEVABLE_TOLERANCE_AND_FITS_BY_PROCESS"]},
    {"id": "STD_ISO_2768", "designation": "ISO 2768", "title": "General tolerances",
     "held_document_id": None, "requested_by": ["consolidated review section 10.4 item 8",
                                                 "docs/LITERATURE_REQUESTS_HARDWARE_LOOP.md H6"],
     "capability_ids": ["CAP_LITREQ_DOCS_LITERATURE_REQUESTS_HARDWARE_LOOP_MD_H6_ACHIEVABLE_TOLERANCE_AND_FITS_BY_PROCESS"]},
    {"id": "STD_ASTM_E606", "designation": "ASTM E606", "title": "Strain-controlled fatigue testing",
     "held_document_id": None, "requested_by": ["consolidated review section 10.4 item 8"], "capability_ids": []},
    {"id": "STD_JCGM_GUM", "designation": "JCGM uncertainty guides (GUM, ISO/IEC Guide 98-3)",
     "title": "Guide to the expression of uncertainty in measurement",
     "held_document_id": None, "requested_by": ["consolidated review section 10.4 item 8",
                                                 "docs/LITERATURE_REQUESTS_HARDWARE_LOOP.md H4"],
     "capability_ids": ["CAP_LITREQ_DOCS_LITERATURE_REQUESTS_HARDWARE_LOOP_MD_H4_MEASUREMENT_UNCERTAINTY_FRAMEWORK"]},
    {"id": "STD_ASME_PTC_19_1", "designation": "ASME PTC 19.1", "title": "Test Uncertainty",
     "held_document_id": None, "requested_by": ["docs/LITERATURE_REQUESTS_HARDWARE_LOOP.md H4 alternative"],
     "capability_ids": ["CAP_LITREQ_DOCS_LITERATURE_REQUESTS_HARDWARE_LOOP_MD_H4_MEASUREMENT_UNCERTAINTY_FRAMEWORK"]},
)

SCOPE = ("Source availability only. HELD means an exact current corpus PDF is mapped; it is not compliance, "
         "rule extraction or applicability. BLOCKED_ON_STANDARD keeps dependent rules unavailable.")


def harvest() -> dict:
    docs = store.index_by("manifest")
    caps = store.index_by("capabilities")
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for entry in REGISTRY:
        doc = docs.get(entry["held_document_id"] or "")
        held = bool(doc and doc.get("sha256") and (doc.get("page_count") or 0) > 0)
        row = {"id": entry["id"], "type": "StandardSource", "designation": entry["designation"],
               "title": entry["title"], "requested_by": entry["requested_by"],
               "capability_ids": [c for c in entry["capability_ids"] if c in caps],
               "missing_capability_ids": [c for c in entry["capability_ids"] if c not in caps],
               "state": "HELD" if held else "BLOCKED_ON_STANDARD",
               "edition": entry.get("held_edition") if held else "UNSPECIFIED",
               "held_document_id": entry["held_document_id"] if held else None,
               "held_source_sha256": doc.get("sha256") if held else None,
               "rule_extraction_state": "NOT_EXTRACTED", "compliance_state": "NOT_ASSESSED",
               "provenance_kind": "PROJECT_DERIVES", "scope": SCOPE, "harvested_at": now}
        if entry["held_document_id"] and not held:
            row["held_document_state"] = "MAPPED_DOCUMENT_MISSING_OR_EMPTY"
        rows.append(row)
    store.save("standards", rows, meta={"generated_at": now, "scope": SCOPE, "n": len(rows)})
    return {"standards": len(rows), "held": sum(r["state"] == "HELD" for r in rows),
            "blocked_on_standard": sum(r["state"] == "BLOCKED_ON_STANDARD" for r in rows)}


def graph_records() -> tuple[list[dict], list[dict]]:
    nodes, edges = [], []
    for r in store.load("standards"):
        nodes.append({"id": r["id"], "type": "StandardSource", "label": r["designation"], "state": r["state"],
                      "edition": r["edition"], "rule_extraction_state": r["rule_extraction_state"]})
        if r.get("held_document_id"):
            edges.append({"from": r["id"], "to": r["held_document_id"], "type": "HELD_AS",
                          "provenance": "PROJECT_DERIVES:kg.standards"})
        for cap in r.get("capability_ids", []):
            edges.append({"from": cap, "to": r["id"], "type": "BLOCKED_ON" if r["state"] != "HELD" else "CITES",
                          "provenance": "PROJECT_DERIVES:kg.standards"})
    return nodes, edges
