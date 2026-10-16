# LREKit local scientific memory

This repository-local system connects source PDFs and passages to equations,
claims, model-registry entries, code, tests, requirements, and provenance.
It runs offline using JSON stores, SQLite FTS5, and a NumPy/SciPy LSA vector index.
The retrieval API is a Python module; there is no HTTP service.

## Start here

The wrapper tries `.venv`, `.venv-jax`, then `.venv-kg` (NumPy and SciPy are required). Set
`KG_PYTHON=/path/to/python` to use another environment. PDF conversion additionally
uses `.venv-kg` with PyMuPDF, pymupdf4llm, and optional Tesseract OCR.
The PDF corpus is local and gitignored; a fresh checkout does not include it.

```bash
knowledge/kg.sh status
knowledge/kg.sh ask "What supports the Rao variational nozzle method?"
knowledge/kg.sh --json context --task "Review coolant heat-transfer validity"
knowledge/kg.sh equation "Rao"
knowledge/kg.sh basis raosim/physics.py
knowledge/kg.sh sources EQ_ID
knowledge/kg.sh artifacts "faceplate"
knowledge/kg.sh artifact CAD_ID
```

Use the concrete equation ID returned by a query in place of `EQ_ID`.
`--json` belongs before the subcommand. For Python callers:

```bash
PYTHONPATH=knowledge python -c 'from kg.api import retrieve_for_task; print(retrieve_for_task("Rao nozzle"))'
```

Read [AGENTS.md](AGENTS.md) for the authority and provenance contract and
[the implementation audit](reports/IMPLEMENTATION_AUDIT_2026-09-20.md) for what
Claude completed, what was repaired, and the remaining implementation scope.
[The generated report](reports/latest_report.md) contains current store counts.

## Refresh without losing completed reviews

```bash
knowledge/kg.sh sync
knowledge/kg.sh sync --convert   # only when PDFs are missing or stale
knowledge/kg.sh rebuild
knowledge/kg.sh --json validate
```

Rebuild performs sync, manifest/chunk/vector refresh, registry and repository
harvest, requirements, evidence freshness checks, graph assembly, validation,
engineering report/CAD observations, and reports. It preserves extracted evidence and source-support reviews.
Changed or removed evidence is marked `STALE_SOURCE` and excluded from normal
scientific retrieval until reviewed again. Historical records remain inspectable.
The vector index is rebuilt only when its corpus fingerprint or artifacts change.

Missing vector artifacts produce an explicit lexical-only fallback. Missing
corpus material cannot be recovered from a checkout: supply the local source
library before rebuilding. Retrieval does not call remote models or spend tokens.
Long-lived Python clients should call `kg.api.reset_caches()` after an external
process updates stores.

## Resume extraction in small steps

```bash
knowledge/kg.sh --json pending --limit 3
knowledge/kg.sh --json pending --subsystem nozzle --limit 1
knowledge/kg.sh --json packet MODEL_NOZZLE_RAO_TOP_BEZIER --max-chars 12000
```

`pending` writes the full queue to `reports/extraction_queue.json`; the CLI shows
only the requested number of items. It skips completed source reviews and lists
verification work separately. A bounded packet contains candidate passages,
source hashes and PDF locations. It is not an exhaustive search of the source.
These commands never spawn agents.

Review one source, emit a result following
[workflows/extraction.schema.json](workflows/extraction.schema.json), and save it
as one JSONL line: `{"type":"result","key":"a-stable-key","value":{...}}`.
Include the packet's source SHA-256. A real example is
[the Rao review](workflows/results/rao1958-2026-09-20.jsonl).

```bash
.venv/bin/python knowledge/tools/bank_extraction.py path/to/results.jsonl
```

Banking checks contiguous source anchors and rejects a supplied hash for a
replaced PDF. Replaying an already banked result does not inflate counts or
erase independent review. A located quote proves its location, not the accuracy
of an equation transcription or interpretation. Independent verification is a
separate step using `kg.harvest_extraction.apply_verification`, followed by
`knowledge/kg.sh rebuild`.

## Engineering report and CAD evidence

`artifacts.json` holds observations from existing `design_report_v2.json` and
`injector_manufacturing_report.json` files under local `builds/`. Rebuild discovers
these recognized formats, including ignored build outputs. `find_artifacts()` and
`get_artifact()` expose records and provenance; task context includes matching
`engineering_artifacts`. `sources ARTIFACT_ID` resolves the report and JSON locator.

An `EngineConfiguration` is a serialized input or injector-layout snapshot with
an explicit scope and reproducible payload selection. Its hash is not a producer
configuration digest. CAD records distinguish declared paths, observed byte
hashes, unverified stored inspections, and unknown generation-time integrity.
`BOUND_TO_CONFIGURATION` records a report association with `binding_state=UNKNOWN`.
Changed or missing evidence retains inactive history and loses active graph links.
Manufacturing records describe reference plans; they do not establish execution.
No experiments, measurements or calibrations are inferred from simulations.

## PDF conversion quality

OCR eligibility is now per page, including sparse pages in otherwise readable
documents. OCR output remains low confidence. Damaged embedded text, equations,
tables and reading order need separate review; a `needs_review` flag does not mean
the entire PDF needs OCR. A checkpointed diagnostic samples native text only:

```bash
.venv-kg/bin/python knowledge/tools/triage_pdfs.py --output knowledge/reports/pdf_triage.json
```

Each PDF runs in an isolated worker with a timeout; completed results resume by
source checksum. See [the local parser recommendation](reports/PDF_PARSING_2026-10-14.md)
for the initial parser recommendation. A bounded local Docling pilot now preserves
raw JSON, table cells, formula candidates, figure locators, model revisions and
source hashes alongside the reviewed Markdown. It does not replace that Markdown.
The pilot found adjacent-column crop contamination and scan-symbol errors; the
original PDF and independent review decide which transcriptions enter the science
stores. Replacing reviewed Markdown still requires re-grounding affected evidence.

```bash
uv venv --python 3.12 .venv-docling
uv pip install --python .venv-docling/bin/python -r knowledge/requirements-docling.lock
.venv-docling/bin/python knowledge/tools/docling_enrich.py DOC_ID --pages 6:8
knowledge/kg.sh --json enrichments --document-id DOC_ID
knowledge/kg.sh --json coverage --documents
.venv-jax/bin/python knowledge/tools/docling_batch.py           # full corpus plan
.venv-jax/bin/python knowledge/tools/docling_batch.py --run --max-jobs 1 --pages-per-job 2
```

Each bounded batch job banks before the next; timeout/failure journals and source
checksums support resumption. First use downloads local model weights; document
inference uses no remote service. Formula-region re-enrichment is available in
`tools/docling_formula_regions.py` when automatic layout splits a formula. Region
boxes must be located visually on the original page. Model output remains an
unverified derivative even when it is LaTeX. Reviewed corrections may explicitly
supersede incomplete same-source/same-number transcriptions, retaining history.

Registry promotion/basis hints are labeled heuristics. `execution_reach` reports
bounded static call-site evidence separately from runtime traces and optimizer
leverage. Resolved registry/constraint citations use `CITES`; they do not establish
page-verified `SUPPORTED_BY` evidence. Current implementation locators and body
fingerprints are refreshed without certifying their scientific comparisons.

## Verification

```bash
.venv/bin/python -m pytest knowledge/tests -q
```

When `.venv` is absent but `.venv-jax` exists, use
`.venv-jax/bin/python -m pytest knowledge/tests -q` (the verified checkout's fallback).

The corpus-free regression modules cover preservation, freshness, search,
workflow replay and queue behavior. The original corpus acceptance tests run
when the local chunk database is present. Broader LREKit simulation tests remain
separate; knowledge-system tests do not physically validate engineering models.
