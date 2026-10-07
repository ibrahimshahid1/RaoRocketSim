# LREKit local scientific memory

This repository-local system connects source PDFs and passages to equations,
claims, model-registry entries, code, tests, requirements, and provenance.
It runs offline using JSON stores, SQLite FTS5, and a NumPy/SciPy LSA vector index.
The retrieval API is a Python module; there is no HTTP service.

## Start here

Use the repository's existing `.venv` (NumPy and SciPy are required). Set
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
and reports. It preserves extracted evidence and source-support reviews.
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

## Verification

```bash
.venv/bin/python -m pytest knowledge/tests -q
```

The corpus-free regression modules cover preservation, freshness, search,
workflow replay and queue behavior. The original corpus acceptance tests run
when the local chunk database is present. Broader LREKit simulation tests remain
separate; knowledge-system tests do not physically validate engineering models.
