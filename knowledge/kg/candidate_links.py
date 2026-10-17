"""Identity checks for reviewed source objects linked to raw parser candidates."""

FIELDS = {"enrichment_candidate_ids": "DERIVED_FROM", "corrects_candidate_ids": "CORRECTS"}


def link_problem(candidate: dict, *, document_id: str, page: int, source_sha256: str,
                 reviewer: dict | None, correction: bool) -> str | None:
    """A rejected candidate can remain correction history, never source support.

    A signed original-page extraction may correct a candidate before the same
    packet's parser-QA file is applied. No parser verdict is inferred here.
    """
    if not candidate or candidate.get("candidate_kind") not in {"formula", "table", "figure"}:
        return "UNKNOWN_PARSER_CANDIDATE"
    if candidate.get("document_id") != document_id:
        return "CANDIDATE_DOCUMENT_MISMATCH"
    if not source_sha256 or candidate.get("source_sha256") != source_sha256:
        return "CANDIDATE_SOURCE_MISMATCH"
    if candidate.get("source_state") != "CURRENT":
        return "CANDIDATE_SOURCE_NONCURRENT"
    pages = candidate.get("pages")
    if (not isinstance(pages, list) or type(page) is not int or page not in pages
            or candidate.get("location_state") == "UNRESOLVED_PAGE"):
        return "CANDIDATE_PAGE_MISMATCH"
    rejected = candidate.get("review_state") == "REJECTED"
    if rejected and not correction:
        return "REJECTED_CANDIDATE_IS_CORRECTION_HISTORY_ONLY"
    if not candidate.get("active", True) and not (correction and rejected):
        return "CANDIDATE_INACTIVE"
    identity = reviewer or {}
    if identity.get("role") not in {"F", "A"} or not str(identity.get("model") or "").strip():
        return "CANDIDATE_LINK_REQUIRES_ORIGINAL_PAGE_REVIEWER"
    return None
