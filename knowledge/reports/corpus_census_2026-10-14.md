# Corpus census — 2026-10-14

This is a mechanical audit snapshot, not a certification of mathematical, physical, or extraction completeness. The JSON contains every payload record and every conversion source; the CSV is the per-document coverage matrix. No knowledge stores or source documents were modified.

## Identity and indexing

- 286 physical PDFs plus 6 archived PDF members; 286 conversion rows; 252 distinct payload records.
- 248 documents selected for indexing; 4 alternate payloads intentionally suppressed by bibliographic deduplication; 247 canonical documents have actual chunks. The 248 SQLite document-state rows include one zero-chunk document; existing status reports this row as chunked.
- 0 current source SHA mismatches, 0 unavailable sources, 0 page-count mismatches, 0 missing derivatives.
- Six physical PDF paths have no separate conversion row, but all six were SHA-verified as byte-identical aliases of tracked archive members. Zero document payloads are missing from the conversion manifest. There are 252 unique payloads across physical and archived copies; input stores changed during census: none.

The unchunked source `propulsion_texts/10.1007_978-3-030-14788-4.pdf` opens as a zero-page PDF. Its current derivative carries no PDF content. Original-source restoration is required; enrichment cannot recover pages that are absent from the payload.

## Extraction and verification coverage

- Only 8 canonical documents have structured source objects; 240 have none. 240 have no structured equation records.
- 45 equations, 186 claims, 33 assumptions, 52 empirical constants, 36 source-stated validity records.
- 9 equations and 5 claims are independently CONFIRMED; 217 remain unverified. Confirmed equations occur in 1 document(s).
- 7 equations have no model_ids. An intentional generic concept is possible; review the listed IDs before creating a scientific support edge.
- Canonical conversion states: {'needs_review': 230, 'good': 18}. 220 documents have historical native equation signals and no structured equations. 149 have more native than Markdown signals. 108 contain source table captions and zero Markdown tables.

Equation signals are heuristic line matches and can include prose, symbols and tables. The historical native/Markdown signal difference does **not** count missing equations. Correct completeness remains NOT_ESTABLISHED for every document; no denominator of all actual source equations was measured.

## Integration gaps

- Model support states: {'PARTIALLY_SUPPORTED': 9, 'NOT_FOUND_IN_SOURCE': 1, 'SUPPORTED': 2, 'NOT_CHECKED': 49}; a registry citation is not source support.
- Legacy paper indexes contain 224 rows and 224 extraction reports, versus 286 conversion sources; 67 canonical documents are absent by source/alternate path. Legacy extraction reports label heuristic signals as `Equations`; these are not stored, page-verified Equation objects.
- 3 paper notes are registered; all source-Markdown and note hashes match their note manifest. Refresh or explicitly label them stale only when their sources change.

Orphan Markdown derivatives/concept files:

- `propulsion_texts/propulsion_texts_for_agents/markdown/grisson1991_AEDC-TR-91-1.md`
- `propulsion_texts/propulsion_texts_for_agents/markdown/pintle_injector/PINTLE_LITERATURE_CATALOG.md`

These files have no conversion-manifest source mapping. A human-authored catalog should be classified as a catalog rather than assigned an invented PDF. The Grissom derivative requires original-source identification and payload verification before ingestion.

## Bounded remediation order

1. Preserve the complete source/derivative identity mapping and note the four alternate payloads; never equate duplicate suppression with missing extraction.
2. Use Docling enrichment on a bounded source window with native/Markdown damage or an implementation-critical missing formulation. Preserve original quotes, page anchors, source/derivative hashes, equation notation, assumptions, applicability and verification states.
3. Bank source objects and independent checks separately; do not bulk-convert candidate signals into scientific assertions.
4. Restore the zero-page original source and distinguish actual nonempty chunk coverage from the presence of a document-state row.
5. Review unchecked and partial model support in the existing queue, then link only formulations and assumptions actually supported by original PDF pages.
6. Refresh paper notes only when changed; classify orphan catalogs; reproduce source identity for the Grissom Markdown before treating it as source evidence.
7. Repeat this census after enrichment/rebuild. Full per-document/per-page completeness is outstanding and cannot be inferred from successful indexing.
