# Local PDF parsing recommendation

User preference: local tools first; cloud only if necessary. This recommendation
is based on the corpus's observed problems and current official tool capabilities.
It is not the result of a Docling/Marker accuracy benchmark on these PDFs.

## Recommended route

Keep native text extraction for clean prose. Pilot **Docling with formula
enrichment** on difficult scientific pages: its formula stage emits LaTeX, while
its document representation retains structure, reading order, provenance and
available bounding boxes. Formula enrichment requires explicit activation and
adds computation. Those properties make it a suitable local baseline for this
memory layer. [Formula enrichment](https://docling-project.github.io/docling/usage/enrichments/),
[document representation](https://docling-project.github.io/docling/concepts/docling_document/).

Compare **Marker** on the same pages as the second local candidate. It supports
equations, tables and CPU/MPS execution. Its Apple Silicon path currently needs
llama.cpp's local server. Optional `--use_llm` can invoke a configured external
provider, so leave that option off for a local-only comparison.
[Official Marker documentation](https://github.com/datalab-to/marker).

These are layout- and math-aware pipelines; OCR is one component. Re-running plain
OCR over every page does not establish correct subscripts, signs, units, table
alignment or reading order. No cloud converter or paid API has been used.

## Concrete corpus evidence and repairs

The corpus has 248 canonical publications. Most conversion flags are heuristic
review requests, not a measured count of papers needing OCR. The 14-page electric
pump paper has readable prose but damaged interleaved two-column equations and
tables. Original PDF pages 6-8 and 10-12 were visually checked for the source
review; Eq. 29's power convention would be consequential if misread.

The converter previously skipped minority scanned pages because it required at
least half the document to be text-sparse. OCR now operates per sparse page,
restores omitted page markers, and flags OCR-backed output as low confidence.
Legacy scanned rows are also treated as low confidence even if an older metadata
flag said otherwise. Existing derivatives were not overwritten by this repair.

`pdf_triage_2026-10-14.json` samples first/middle/last native text with isolated
per-PDF timeouts and checksum checkpoints. Blank covers and figure-only pages
can be text-sparse; abundant damaged OCR can pass the character threshold.
Its diagnostics identify review candidates, not a complete OCR-needed census.
The completed sample covers 724 pages across all 248 canonical documents: 25
text-sparse sampled pages across 20 documents, one document with sampled replacement
characters, and no remaining worker errors/timeouts. Six archive-only source PDFs
were restored to their recorded paths after exact SHA-256 verification; indexed
text and scientific object IDs were preserved.

## Bounded comparison before adoption

Use the same 12-20 pages for both parsers, with original PDFs as ground truth:

| Material | Candidate pages | Check |
|---|---|---|
| Rao 1958 mathematics | PDF pp. 2-4 | Signs, powers, limits, subscripts, equation numbers |
| Lee electric-pump paper | PDF pp. 6-8, 11-12 | Two-column order, power conventions, table values and units |
| NASA pump criteria | SP-8109 selected equation/table pages after packet retrieval | Scan quality, units, empirical scope |
| Sparse/failed diagnostic candidates | From the triage report, after visual classification | Real scans versus blank/figure pages; parser timeout recovery |

Score exact equation transcription and table cells, missing passages, page
locators, runtime and memory. Text length and vendor benchmark rankings alone
cannot choose the best parser for this corpus. Preserve source SHA-256, original
page numbers, bounding boxes where available, parser/model versions and raw
structured output. Keep candidate output beside the existing derivative until
reviewed. A parser change must trigger source re-grounding rather than preserve
old scientific verification automatically.
