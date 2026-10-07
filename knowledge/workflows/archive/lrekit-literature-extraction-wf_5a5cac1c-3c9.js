export const meta = {
  name: 'lrekit-literature-extraction',
  description: 'Extract page-anchored equations, claims, assumptions and validity statements from load-bearing LREKit sources',
  phases: [
    { title: 'Model-driven', detail: 'one agent per model_registry entry: find its supporting equations in the cited source' },
    { title: 'Document-driven', detail: 'one agent per 12-page window of each load-bearing paper' },
  ],
}

const ARGS_PATH = args.path
const N_MODELS = args.n_models
const N_WINDOWS = args.n_windows

const VAR = {
  type: 'object',
  properties: {
    symbol: { type: 'string' }, meaning: { type: 'string' }, units: { type: 'string' },
    role: { type: 'string', enum: ['input', 'output', 'parameter', 'constant', 'unknown'] },
  },
  required: ['symbol', 'meaning'],
}
const EQUATION = {
  type: 'object',
  properties: {
    label: { type: 'string' },
    equation_number: { type: 'string' },
    form: { type: 'string' },
    description: { type: 'string' },
    variables: { type: 'array', items: VAR },
    assumptions: { type: 'array', items: { type: 'string' } },
    validity: { type: 'array', items: { type: 'string' } },
    page: { type: 'integer' },
    quote: { type: 'string' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    extraction_note: { type: 'string' },
  },
  required: ['label', 'form', 'page', 'quote', 'confidence'],
}
const CLAIM = {
  type: 'object',
  properties: {
    claim_type: { type: 'string', enum: ['definition', 'theoretical', 'empirical', 'methodological', 'comparative', 'assumption', 'constraint', 'recommendation', 'validity_statement', 'design_criterion', 'derived_result', 'benchmark_result'] },
    subject: { type: 'string' }, predicate: { type: 'string' }, object: { type: 'string' },
    statement: { type: 'string' },
    qualifiers: { type: 'array', items: { type: 'string' } },
    page: { type: 'integer' },
    quote: { type: 'string' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
  },
  required: ['claim_type', 'subject', 'predicate', 'object', 'statement', 'page', 'quote', 'confidence'],
}
const BOUND = { type: 'object', properties: { variable: { type: 'string' }, lower: { type: 'string' }, upper: { type: 'string' }, units: { type: 'string' } }, required: ['variable'] }
const VALIDITY = {
  type: 'object',
  properties: {
    statement: { type: 'string' }, applies_to: { type: 'string' },
    bounds: { type: 'array', items: BOUND },
    page: { type: 'integer' }, quote: { type: 'string' },
  },
  required: ['statement', 'applies_to', 'page', 'quote'],
}
const CONSTANT = {
  type: 'object',
  properties: { name: { type: 'string' }, value: { type: 'string' }, units: { type: 'string' }, context: { type: 'string' }, page: { type: 'integer' }, quote: { type: 'string' } },
  required: ['name', 'value', 'page', 'quote'],
}
const EXTRACT_SCHEMA = {
  type: 'object',
  properties: {
    doc_id: { type: 'string' },
    target: { type: 'string' },
    equations: { type: 'array', items: EQUATION },
    claims: { type: 'array', items: CLAIM },
    validity_statements: { type: 'array', items: VALIDITY },
    empirical_constants: { type: 'array', items: CONSTANT },
    assumptions: { type: 'array', items: { type: 'object', properties: { statement: { type: 'string' }, page: { type: 'integer' }, quote: { type: 'string' } }, required: ['statement', 'page', 'quote'] } },
    supports_model: {
      type: 'object',
      properties: {
        found: { type: 'boolean' },
        equation_labels: { type: 'array', items: { type: 'string' } },
        mismatch_notes: { type: 'string' },
        state: { type: 'string', enum: ['SUPPORTED', 'PARTIALLY_SUPPORTED', 'NOT_FOUND_IN_SOURCE', 'SOURCE_UNREADABLE'] },
      },
      required: ['found', 'state'],
    },
    notes: { type: 'string' },
  },
  required: ['doc_id', 'target', 'equations', 'claims'],
}

const RULES = `
You are extracting scientific objects from ONE propulsion source document rendered as Markdown for the LREKit knowledge system.
The Markdown is a derivative of a PDF. Page markers look like <!-- PDF page N -->. Some files begin with a
[!WARNING] callout meaning the conversion is low-confidence.

NON-NEGOTIABLE RULES
1. Every equation, claim, validity statement, assumption and constant MUST carry a 'quote': 60-240 characters
   copied VERBATIM from the Markdown (copy-paste; do not retype, normalise, translate symbols, or fix OCR).
   A later automated step searches the corpus for your quote; if it is not found the object is discarded.
2. 'page' is the PDF page number from the nearest preceding page marker above the quoted text.
3. Transcribe equations in the source's own notation. Never modernise symbols, never substitute a form you know
   from elsewhere, never "repair" a garbled equation. If garbled, set confidence=low and describe in extraction_note.
4. Record the assumptions and validity limits the SOURCE states next to the equation. Do not add your own.
   If the source states no validity range, leave 'validity' empty - do not invent one.
5. Do not report facts you know from other literature. Only what THIS document states.
6. Prefer fewer, exactly-anchored objects over many loosely-anchored ones. Empty arrays are acceptable.
7. Claims are scientific assertions worth retrieving later (definitions, empirical findings, design criteria,
   recommendations, comparisons, validity statements). Not every sentence is a claim.
8. Use Read with offset/limit or Grep on the Markdown file. Large books: use Grep to locate the relevant
   section by the terms in the registry hint before reading.
`

phase('Model-driven')
const modelIdx = Array.from({ length: N_MODELS }, (_, i) => i)
const modelResults = await parallel(modelIdx.map(i => () =>
  agent(`${RULES}

TASK (model-driven): Read the JSON file ${ARGS_PATH} and take model_targets[${i}].
It names a raosim.model_registry entry (model_id, registry_id, relation, equation_ref, validity_text, cited_source_text)
and the corpus document LREKit claims supports it (doc_id, markdown path, page_count).

Open the Markdown at the given path. Locate the passage(s)/equation(s) in THIS document that correspond to the registry's
'relation' and 'equation_ref'. Extract them (equations, claims, validity statements, assumptions, constants) with verbatim quotes.
Then fill supports_model honestly:
 - SUPPORTED if the document states the relation the registry attributes to it;
 - PARTIALLY_SUPPORTED if only part is there or the form/assumptions differ (explain in mismatch_notes);
 - NOT_FOUND_IN_SOURCE if you cannot find it after a genuine search (grep several synonyms);
 - SOURCE_UNREADABLE if the rendering is too garbled to tell.
Set target = the model_id. Set doc_id = the doc_id from the target.`,
    { label: `model:${i}`, phase: 'Model-driven', schema: EXTRACT_SCHEMA })
))

phase('Document-driven')
const winIdx = Array.from({ length: N_WINDOWS }, (_, i) => i)
const windowResults = await parallel(winIdx.map(i => () =>
  agent(`${RULES}

TASK (document-driven): Read the JSON file ${ARGS_PATH} and take windows[${i}].
It gives doc_id, markdown path, page_start, page_end. Read ONLY the Markdown between the marker
<!-- PDF page ${'${page_start}'} --> and the marker following page ${'${page_end}'} (use Grep -n "PDF page" to find line
numbers, then Read that line range). Extract every scientifically retrievable equation, claim, validity statement,
assumption and empirical constant in that window, with verbatim quotes and page numbers.
Nomenclature sections: record variables' meanings via the 'variables' of the equations that use them; do not
emit a claim per nomenclature line. Reference lists: skip.
Set target = "pages A-B" and doc_id from the window.`,
    { label: `doc:${i}`, phase: 'Document-driven', schema: EXTRACT_SCHEMA })
))

const failedModels = modelIdx.filter(i => !modelResults[i])
const failedWindows = winIdx.filter(i => !windowResults[i])
log(`model-driven: ${modelResults.filter(Boolean).length}/${N_MODELS}; document-driven: ${windowResults.filter(Boolean).length}/${N_WINDOWS}`)
return {
  model_results: modelResults.map((r, i) => r ? { index: i, ...r } : null).filter(Boolean),
  window_results: windowResults.map((r, i) => r ? { index: i, ...r } : null).filter(Boolean),
  failed_models: failedModels, failed_windows: failedWindows,
}
