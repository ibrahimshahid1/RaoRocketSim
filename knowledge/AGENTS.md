# LREKit scientific knowledge system — agent manual

This directory is LREKit's **local scientific memory and engineering provenance
layer**. Read this file before any task that changes or evaluates physics,
equations, correlations, validity criteria, empirical coefficients, structures,
injectors, sprays, pumps, optimisation constraints, CAD dimensions derived from
physics, manufacturing rules, requirements, measurement comparison, calibration,
validation, or qualification claims.

It is an **index and reasoning aid**, not a replacement for reading the source.
The graph never outranks the PDF it points to.

---

## 0. Startup protocol (mandatory for scientifically consequential work)

```bash
knowledge/kg.sh status                       # 1. corpus synchronized? stores populated?
knowledge/kg.sh context --task "<your task>" # 2. retrieve_for_task(): the context package
knowledge/kg.sh model  "<concept>"           # 3. what LREKit already implements, and on what basis
knowledge/kg.sh equation "<concept>"         # 4. extracted, page-anchored equations
knowledge/kg.sh claims "<subject>"           # 5. claims, incl. validity statements and assumptions
knowledge/kg.sh validity MODEL_...           # 6. validity envelopes (prose + any source-stated bounds)
knowledge/kg.sh basis raosim/<file>.py:<Symbol>   # 7. the scientific basis of existing code
knowledge/kg.sh sources <EQ_|CLM_|VAL_ id>   # 8. provenance + source context + PDF page to escalate to
knowledge/kg.sh compare "<topic>"            # 9. competing formulations across documents
```

Then: **open the PDF page** for every load-bearing value, equation or limit
(`sources` prints the path and page). Distinguish literature-derived science from
implementation choices (every model and constant carries a `basis_class`). Only
then implement. This applies to auditing/repairing existing code too — **existing
code is not scientific authority merely because it exists**.

If `status` reports `synchronized: false`, run the sync (§3) before trusting
retrieval; a PDF that is on disk but not converted/ingested is invisible to the
system and its absence from results means nothing.

Purely mechanical tasks (formatting, renames, packaging) do not need retrieval.

---

## 1. What is here

```
knowledge/
  AGENTS.md            this manual
  schema.json          ontology (node/edge types, provenance kinds, state vocabularies) — versioned
  manifest.json        one Document per unique PDF payload: bibliographic data, sha256, conversion state,
                       generated-Markdown path, alternate copies, known limitations
  quantities.json      canonical physical quantities: symbols, aliases, units, dimension, ambiguity notes
  models.json          the 61 raosim.model_registry entries as Model nodes (+ source-support state from extraction)
  validity.json        validity envelopes: registry prose + source-stated statements/bounds (page-anchored)
  equations.json       page-anchored equations extracted from the literature (verbatim quote, grounded)
  claims.json          page-anchored claims (definitions, empirical results, criteria, validity statements...)
  entities.json        assumptions and literature-stated empirical constants (page-anchored)
  software_links.json  SoftwareComponent nodes (file:line verified), hard-coded constants with basis_class,
                       refusal conditions, and SUPPORTED_BY / CITES links
  tests.json           tests classified by verification kind (conservatively) + benchmark links
  capabilities.json    explicit UNAVAILABLE / NOT_IMPLEMENTED / BLOCKED_ON_LITERATURE paths and literature requests
  repo_documents.json  docs/*.md with dates, STATUS heads, precedence relations, staleness signals
  requirements.json    requirement / optimizer-constraint / postsolve-gate / diagnostic records (see §7)
  relationships.json   optional cross-store typed edges not owned by a single record; absent means empty
  provenance.json      one ProvenanceRecord per asserted object (who asserts it, where, verified how)
  artifacts.json       observed input/layout, CAD, producer-report and reference-process snapshots
  ingestion_state.json per-document current states (also in reports/latest_ingestion.json)
  graph/graph.json     assembled graph (view over the stores) ; graph.graphml (export, gitignored)
  vector_index/        chunks.sqlite (FTS5) + LSA vectors — DERIVED, gitignored, rebuild with `kg.sh rebuild`
  reports/             latest_sync.json, latest_ingestion.json, latest_validation.json, validation_<ts>.json
  kg/                  python package (api, cli, index, graph, ingest, chunker, ground, harvest_*, validate)
  tools/               pdf_to_markdown.py, sync_corpus.py, kg_pipeline.py
```

### Corpus locations and authority

| Layer | Path | Authority |
|---|---|---|
| Source PDFs | `propulsion_texts/**/*.pdf` (gitignored, local) | **authoritative** |
| Generated Markdown | `propulsion_texts/propulsion_texts_for_agents/markdown/**.md` | derivative (`GeneratedDocument DERIVED_FROM Document`) |
| Conversion manifest | `propulsion_texts/propulsion_texts_for_agents/_conversion_manifest.json` | machine record of PDF→MD (sha256, converter, status, issues) |
| Human-authored corpus notes | `propulsion_texts/propulsion_texts_for_agents/{README,AGENTS,LIBRARY_USAGE}.md`, `notes/`, `PINTLE_LITERATURE_CATALOG.md` | human-authored; never overwritten by conversion |
| Repository documents | `docs/*.md`, `README.md` | **implementation-state** authority only, not scientific authority |
| Model registry | `raosim/model_registry.py` | the repo's own provenance statement (`PROJECT_DERIVES`) |

Generated Markdown produced by this system carries an `<!-- lrekit-kg-conversion ... -->`
block (source path, sha256, converter, timestamp, status). Files without it were produced by
the earlier corpus pass (`pymupdf4llm-classic`) and are registered via the conversion manifest.
Markdown that begins with `> [!WARNING]` is low-confidence: verify against the PDF.

---

## 2. Provenance model

Every stored scientific object has a `provenance_id` → `provenance.json` record with `kind`:

| kind | meaning |
|---|---|
| `SOURCE_STATES` | the cited document says it, on that page; the `quote` was located verbatim in the indexed text (`grounded`: `GROUNDED` or `GROUNDED_PAGE_SHIFTED` with the corrected page) |
| `PROJECT_DERIVES` | a repository document or the model registry asserts it |
| `CODE_STATES` | LREKit source code asserts it (docstring/comment/registry id); file:line verified against the working tree |
| `SYSTEM_INFERS` | this knowledge system's own terminology/normalisation decision (e.g. quantity aliases) — lowest authority |
| `IMPLEMENTATION_CHOICE` / `EMPIRICALLY_CALIBRATED` | as named |

Anything an LLM extracted but could not anchor verbatim in the source text was **not stored**;
it is listed in `reports/latest_extraction.json` under `ungrounded`. Missing information stays missing.

`basis_class` on models and hard-coded constants: `LITERATURE_DERIVED`, `PROJECT_DERIVED`,
`IMPLEMENTATION_CHOICE`, `EMPIRICALLY_CALIBRATED`, `REPOSITORY_HEURISTIC`, `UNKNOWN_BASIS`.
A constant with `UNKNOWN_BASIS` is exactly that; do not promote it.

Models also carry `source_support_state` from the model-driven extraction pass:
`SUPPORTED`, `PARTIALLY_SUPPORTED` (read `source_support_notes` — the form or assumptions
differ from what the registry says), `NOT_FOUND_IN_SOURCE` (the citation could not be confirmed
in the local copy), `SOURCE_UNREADABLE`, or absent (not yet checked).

---

## 3. Corpus synchronisation (PDF → Markdown)

```bash
knowledge/kg.sh sync              # scan only: missing / stale / orphan / duplicate payloads
knowledge/kg.sh sync --convert    # convert missing+stale PDFs (uses .venv-kg: pymupdf4llm + tesseract)
knowledge/kg.sh rebuild           # manifest -> chunks -> vectors -> registry/repo harvest -> graph -> validation
```

Mapping evidence, strongest first: manifest sha256 → provenance block sha256 → manifest path →
filename (never used alone; a filename-only match is treated as **stale** and regenerated).
Identical payloads under different names are one Document with `alternate_paths`; different
payloads with the same DOI / NTRS accession / ISBN are one Document plus `alternate_copies`
(only the better-converted copy is chunked). Conversion never invents text: pages that yield
nothing are emitted as empty page markers and the row records `partial_conversion` or
`pdf_conversion_failed`. Scanned pages are OCR'd (tesseract) and flagged.

When you add a PDF: drop it under `propulsion_texts/`, run `sync --convert`, then `rebuild`.
For the paper to contribute equations/claims (not just retrievable chunks) run the extraction
workflow for it (§9) and `kg.harvest_extraction`.

---

## 4. Retrieval

Hybrid: SQLite FTS5 (BM25, exact symbols like `c*`, `T_wg`) fused with a corpus-wide LSA space
(TF-IDF → truncated SVD, 256 dims) by reciprocal-rank fusion; queries are expanded with the
symbols/aliases of any canonical quantity they mention. Chunks respect document structure
(page markers, headings, paragraphs, atomic tables) and keep `doc_id`, pages, section path,
order and char offsets, so every hit escalates to a PDF page.

Programmatic API (`knowledge/kg/api.py`, stable contract):

```
search_documents(q)  semantic_search(q)  get_document(id)  get_source_context(chunk_id)
find_entity(name)  get_entity(id)  get_neighbors(id)  find_path(a, b)
find_equations(concept)  get_equation(id)
find_claims(subject, predicate=None, obj=None)  get_claim_sources(id)
find_models(concept)  get_model(id)  find_validity_conditions(model)
find_requirements(q)  compare_sources(q)
find_artifacts(q)  get_artifact(id)
find_implementation(scientific_id)  find_scientific_basis(software_ref)
retrieve_for_task(task)                      # the primary interface
```

Use from any repo venv: `PYTHONPATH=knowledge python -c "from kg import api; ..."`.

`retrieve_for_task()` returns a JSON package with: `subsystems`, `quantities`,
`source_documents`, `source_context` (chunks), `equations`, `claims`, `assumptions`,
`validity_envelopes`, `models`, `existing_implementations`, `empirical_constants`,
`requirements_and_gates`, `tests`, `benchmarks`, `unavailable_capabilities`,
`repository_documents`, `conflicting_sources`, `implementation_warnings`, and a `state`.
Objects keep their ids — drill down instead of loading the corpus.

---

## 5. States you will see (and must preserve)

Engineering results: `PASS FAIL UNKNOWN UNAVAILABLE NOT_IMPLEMENTED OUTSIDE_VALIDITY
BLOCKED_ON_LITERATURE BLOCKED_ON_STANDARD NOT_QUALIFIED REPORT_ONLY POSTSOLVE_ONLY
CONVERGED_BUT_INVALID NUMERICALLY_FAILED`.

Retrieval results: `NO_RELEVANT_SOURCE_FOUND INSUFFICIENT_SOURCE_CONTEXT CONFLICTING_FORMULATIONS
AMBIGUOUS_SYMBOL LOW_EXTRACTION_CONFIDENCE UNRESOLVED_ENTITY SOURCE_NOT_INGESTED
OUTSIDE_MODEL_VALIDITY SCIENTIFIC_BASIS_UNKNOWN PLANNED_NOT_IMPLEMENTED NO_VALIDITY_RANGE_LOCATED`.

"No supporting equation was found in the ingested corpus" ≠ "no such equation exists".
"No validity range was located" must never become an invented envelope.
"The implementation returns a number" ≠ "the model is valid here".

Symbols are contextual: `resolve_symbol("k")` returns both specific-heat ratio and thermal
conductivity with `AMBIGUOUS_SYMBOL`; never merge on the glyph.

---

## 6. Model variants are not flattened

Registry models record `promotion_status` (`optimizer_coupled`, `postsolve_only`, `report_only`,
`screening`, `reference_oracle`, `planned_not_implemented`, `unavailable`), `validation_level`
(the registry's own wording, e.g. `software_verified_not_cht_validated`), `verification_kinds`,
and `hardware_qualified: false` (never asserted). Software components record `variant_of_id`
with `PARITY_CHECKED_AGAINST` (NumPy oracle ↔ JAX) or `COMPARED_WITH` (traditional ↔ MDO,
TOP ↔ variational). A numerical parity test is not physical validation; tests are classified
conservatively (`unit_test` unless the file cites a corpus PDF and compares to a tolerance, or
its name says parity/regression/CAD/contract).

---

## 7. Requirements, constraints, gates

Distinct kinds (`schema.json → restriction_kinds`): `requirement`, `optimizer_constraint`,
`postsolve_gate`, `diagnostic`, `applicability_gate`, `evidence_gate`, `manufacturing_gate`.
Software components carry their refusal conditions (`REFUSES_WHEN` edges with the actual code
behaviour: raise / return_unavailable / NaN / clamp / warn_only). A required quantity checked
only postsolve is not enforced by the optimizer; `retrieve_for_task` reports both.

---

## 8. Escalating to the original source

```
graph node → claim/equation/validity → provenance record → document + page → PDF
```
`kg.sh sources EQ_...` prints the quote, the chunk with neighbours, and
`escalation: {pdf, pages, markdown}`. Escalate whenever: extraction `confidence` is not high;
the Markdown was flagged low-confidence; notation looks damaged; validity limits matter; you are
introducing a hard-coded constant; a manufacturing/safety criterion is involved; sources disagree.
Render a page with `gs -sDEVICE=png16m -r150 -dFirstPage=N -dLastPage=N -o out.png file.pdf`.

---

## 9. Adding knowledge honestly

* New literature: §3, then run the extraction workflow (the prompt/schema is recorded in
  `reports/latest_extraction.json` under `method`) for the document and merge with
  `kg.harvest_extraction.harvest(...)`; every object must carry a verbatim `quote` + `page`.
* New code: keep `raosim/model_registry.py` current (it is harvested), and reference ids near
  governing equations / limits / constants, e.g.
  `# Scientific basis: model_id=..., equation_id=EQ_..., source=DOC_... (validity in knowledge/)`.
* Never invent citations, coefficients, correlations, material properties, or validity limits.
  If the corpus lacks the basis: record `UNKNOWN_BASIS`/`BLOCKED_ON_LITERATURE`, add a literature
  request in `docs/LITERATURE_REQUESTS*.md` (harvested as Capability nodes), keep the feature
  unavailable.

---

## 10. Known limitations (current build)

* Initial equation transcription used Markdown renderings; OCR/rendering damage is flagged
  (`extraction_note`, `confidence`). Bounded Docling candidates can be corrected only by
  original-PDF comparison with source/page provenance and retained supersession history.
  Parser output is not automatically a verified equation. The PDF is the reference.
* Semantic retrieval is LSA over this corpus (no neural embeddings, no network); paraphrase recall
  is reasonable within propulsion vocabulary, weak for concepts absent from the corpus.
* Literature extraction is partial. Consult `kg.sh pending` and `reports/extraction_queue.json`
  for actual model and document coverage. Planned document windows are not completed reviews.
  Most documents are retrievable as chunks but have no extracted equations/claims yet.
* `repo_documents.json` relations come from explicit precedence language on a single line;
  STATUS heads are quoted verbatim; when two documents disagree the disagreement is preserved.
* Validity bounds are machine-readable only where a source states numbers (`SOURCE_STATED_BOUNDS`);
  registry validity is prose (`NOT_EXTRACTED`).
* Test classification and subsystem tagging are heuristic and labelled as such.
* `vector_index/` and `graph.graphml` are rebuildable and not committed (size).

See `reports/latest_validation.json` for the current issue list and
`reports/latest_ingestion.json` for per-document ingestion states.

## 11. Resuming after interruption

Use `kg.sh pending --limit 3` to select unfinished source reviews. Export one
`kg.sh --json packet MODEL_ID --max-chars 12000`, inspect its source, and bank a
single result before moving on. The portable contract and example live in
`workflows/extraction.schema.json` and `workflows/results/`; see `README.md`.
The archived Claude scripts require a legacy runtime and use broad parallelism;
they are recovery evidence, not the default continuation mechanism.

`SOURCE_STATES` grounding now requires a contiguous normalized quote; token
similarity cannot prove that a source states a proposition. Changed PDFs or
Markdown invalidate dependent evidence as `STALE_SOURCE`, preserving history
while removing active scientific support. Refuted and stale records remain
inspectable by ID. Missing vectors are explicitly reported as a lexical-only
fallback. Model source-support checks and independent verification remain
separate, visible states in task context.

## 12. Engineering artifact observations

Use `kg.sh artifacts "<query>"`, `kg.sh artifact ID`, or `kg.sh sources ID` for
report/configuration/CAD records. Task context also returns `engineering_artifacts`.
Read scope, source locator, binding state, and qualification state before use.
Only two existing producer-report formats are harvested from local `builds/`;
the graph is not an inventory of every possible output file.

Input and injector-layout snapshots are not globally identified engine designs.
Their payload-selection metadata reproduces the snapshot hash from the source
report. Current CAD byte hashes do not prove which generation-time configuration
produced them. Stored topology inspections remain unverified report assertions.
`BOUND_TO_CONFIGURATION` with `binding_state=UNKNOWN` is only a same-report
association. ManufacturingProcess nodes are reference plans with unavailable
execution evidence. Never promote simulated outputs into Measurement, Experiment,
Calibration, executed manufacturing, or qualification records.

Full rebuild marks changed/disappeared snapshots inactive, retaining history and
excluding their old graph associations. Observations are from the last rebuild;
refresh before relying on current artifact existence. Exported stores may reference
ignored local build files which will be unavailable on another checkout.

## 13. Docling derivatives and audit states

`tools/docling_enrich.py` runs local formula/table processing on one bounded source
window. Raw JSON, model versions/revisions, source hashes, original page numbers
and bounding boxes are retained separately from reviewed Markdown. `enrichments.json`
contains parser candidates and typed table/figure observations, not automatic
scientific support. Inspect `kg.sh enrichments` and `kg.sh coverage --documents`.
Use `tools/docling_batch.py` for a checkpointed plan or bounded sequential jobs.
Reject contaminated crops and malformed scan notation; inspect the original PDF.
Explicit source corrections preserve superseded records and parser failures as
history. A processing ledger never certifies all equations/concepts were extracted.

`execution_reach` is bounded AST evidence, with runtime/leverage explicitly unmeasured
unless a separate run supplies evidence. Registry keyword hints remain labeled.
Code locator/body freshness is separate from implementation-versus-literature review.
Historical model reviews without source fingerprints remain visible but do not
establish current support. Missing source/threshold evidence stays unresolved.

## 14. Review freshness and parser lineage

Models may retain a separate `source_reviews` entry for each source document.
The legacy `source_review` remains the primary-source view; consult
`source_review_state.source_reviews` for every retained comparison. Mixed
fidelity is reported as `MIXED`, with the original classifications preserved.
A secondary review must not replace the primary review or certify another
source's equations. Registry, PDF, and implementation changes require renewed
review of the affected source. Non-equivalent formulations and their contextual
claims are citations, not implementation support.

Independent verification binds two different identities: `source_sha256` for
the PDF bytes and `content_sha256` for the assertion inspected. Generate the
latter with `kg.verification_freshness.content_sha256(record)` and include it
in new verdicts. Read `content_verification` in queries: a historical stored
`CONFIRMED` without a content fingerprint is `LEGACY_CONTENT_NOT_BOUND`, not
proof that all current fields were checked. Re-banking changed scientific
content invalidates its earlier verdict while retaining history. Renewed
verification of changed content requires an explicit content hash. Missing
manifest sources cannot establish current confirmation.

Equations, constants, claims, assumptions, and validity statements can retain
`enrichment_candidate_ids` and `corrects_candidate_ids`. New links need a named
original-page reviewer and matching document, source hash, and page. Rejected
parser notation can remain `CORRECTS` history; it cannot be `DERIVED_FROM`
support. Stored historical references are retained when an edge is withheld;
`graph.json` exposes `candidate_link_issues` for review. No candidate link or
parser verdict establishes physical validation.

Parser QA also keeps an assertion history. Replaying an exact historical review
does not replace the latest state. A bounded source packet may span consecutive
pages when an equation's definitions or applicability continue onto the next
page; inspect the full dependency before judging the equation. The session
replay tool's `success` requires both unchanged stores and resolved operations.
