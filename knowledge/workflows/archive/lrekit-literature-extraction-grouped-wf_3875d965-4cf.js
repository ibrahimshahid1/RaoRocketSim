export const meta = {
  name: 'lrekit-literature-extraction-grouped',
  description: 'Topic-grouped extraction of page-anchored equations/claims/validity from load-bearing LREKit sources',
  phases: [{ title: 'Extract', detail: 'one agent per topic group (models + papers of one subsystem)' }],
}

const GROUPS_PATH = args.path
const GROUP_NAMES = args.groups

const VAR = { type: 'object', properties: { symbol: { type: 'string' }, meaning: { type: 'string' }, units: { type: 'string' }, role: { type: 'string', enum: ['input', 'output', 'parameter', 'constant', 'unknown'] } }, required: ['symbol', 'meaning'] }
const EQUATION = { type: 'object', properties: { label: { type: 'string' }, equation_number: { type: 'string' }, form: { type: 'string' }, description: { type: 'string' }, variables: { type: 'array', items: VAR }, assumptions: { type: 'array', items: { type: 'string' } }, validity: { type: 'array', items: { type: 'string' } }, page: { type: 'integer' }, quote: { type: 'string' }, confidence: { type: 'string', enum: ['high', 'medium', 'low'] }, extraction_note: { type: 'string' } }, required: ['label', 'form', 'page', 'quote', 'confidence'] }
const CLAIM = { type: 'object', properties: { claim_type: { type: 'string', enum: ['definition', 'theoretical', 'empirical', 'methodological', 'comparative', 'assumption', 'constraint', 'recommendation', 'validity_statement', 'design_criterion', 'derived_result', 'benchmark_result'] }, subject: { type: 'string' }, predicate: { type: 'string' }, object: { type: 'string' }, statement: { type: 'string' }, qualifiers: { type: 'array', items: { type: 'string' } }, page: { type: 'integer' }, quote: { type: 'string' }, confidence: { type: 'string', enum: ['high', 'medium', 'low'] } }, required: ['claim_type', 'subject', 'predicate', 'object', 'statement', 'page', 'quote', 'confidence'] }
const BOUND = { type: 'object', properties: { variable: { type: 'string' }, lower: { type: 'string' }, upper: { type: 'string' }, units: { type: 'string' } }, required: ['variable'] }
const VALIDITY = { type: 'object', properties: { statement: { type: 'string' }, applies_to: { type: 'string' }, bounds: { type: 'array', items: BOUND }, page: { type: 'integer' }, quote: { type: 'string' } }, required: ['statement', 'applies_to', 'page', 'quote'] }
const CONSTANT = { type: 'object', properties: { name: { type: 'string' }, value: { type: 'string' }, units: { type: 'string' }, context: { type: 'string' }, page: { type: 'integer' }, quote: { type: 'string' } }, required: ['name', 'value', 'page', 'quote'] }
const RESULT = {
  type: 'object',
  properties: {
    doc_id: { type: 'string' }, target: { type: 'string' },
    equations: { type: 'array', items: EQUATION }, claims: { type: 'array', items: CLAIM },
    validity_statements: { type: 'array', items: VALIDITY }, empirical_constants: { type: 'array', items: CONSTANT },
    assumptions: { type: 'array', items: { type: 'object', properties: { statement: { type: 'string' }, page: { type: 'integer' }, quote: { type: 'string' } }, required: ['statement', 'page', 'quote'] } },
    supports_model: { type: 'object', properties: { found: { type: 'boolean' }, equation_labels: { type: 'array', items: { type: 'string' } }, mismatch_notes: { type: 'string' }, state: { type: 'string', enum: ['SUPPORTED', 'PARTIALLY_SUPPORTED', 'NOT_FOUND_IN_SOURCE', 'SOURCE_UNREADABLE'] } }, required: ['found', 'state'] },
    notes: { type: 'string' },
  },
  required: ['doc_id', 'target', 'equations', 'claims'],
}
const RELATION = { type: 'object', properties: { a_label: { type: 'string' }, a_doc_id: { type: 'string' }, b_label: { type: 'string' }, b_doc_id: { type: 'string' }, relation: { type: 'string', enum: ['EQUIVALENT_TO', 'NOTATION_VARIANT', 'APPROXIMATES', 'DIFFERENT_ASSUMPTIONS', 'CONTRADICTS', 'POSSIBLY_EQUIVALENT'] }, note: { type: 'string' } }, required: ['a_label', 'a_doc_id', 'b_label', 'b_doc_id', 'relation'] }
const GROUP_SCHEMA = {
  type: 'object',
  properties: {
    group: { type: 'string' },
    results: { type: 'array', items: RESULT },
    cross_document_relations: { type: 'array', items: RELATION },
    skipped: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
  },
  required: ['group', 'results'],
}

const RULES = `
You are extracting scientific objects for the LREKit knowledge system from a GROUP of related propulsion sources
(one subsystem). Markdown files are derivatives of PDFs; page markers look like <!-- PDF page N -->. A file starting
with a [!WARNING] callout is a low-confidence conversion.

NON-NEGOTIABLE RULES
1. Every equation, claim, validity statement, assumption and constant MUST carry a 'quote': 60-240 characters copied
   VERBATIM from the Markdown (copy-paste; never retype, normalise, translate symbols, or fix OCR). An automated step
   searches the indexed text for your quote; if it is not found the object is discarded.
2. 'page' = the PDF page from the nearest preceding page marker above the quoted text.
3. Transcribe equations in the source's own notation. Never modernise, substitute a form you know from elsewhere, or
   repair a garbled equation. If garbled: confidence=low and say so in extraction_note.
4. Record only assumptions and validity limits the SOURCE states next to the equation. If none, leave empty.
5. Only what THIS document states; nothing from your own knowledge of other literature.
6. Prefer fewer exactly-anchored objects over many loose ones. Empty arrays are acceptable.
7. Use Grep/Read with line ranges. For long books, Grep for the terms in the registry hint first.
8. Because you see several related papers, ALSO report cross_document_relations between equations you extracted
   (same physics written differently, approximations, different assumptions, contradictions). Cite both labels and
   doc_ids. Do not merge them - just relate them.

For each MODEL TARGET (a raosim.model_registry entry with model_id, relation, equation_ref, validity_text and the
document LREKit claims supports it): find the passage(s)/equation(s) in THAT document that correspond to the
registry's relation, extract them, and fill supports_model honestly: SUPPORTED / PARTIALLY_SUPPORTED (explain the
difference in mismatch_notes) / NOT_FOUND_IN_SOURCE (after grepping several synonyms) / SOURCE_UNREADABLE.
One RESULT per model target with target = model_id.

For each WINDOW (doc_id, markdown, page_start, page_end): read ONLY the Markdown between the marker for page_start
and the marker after page_end (Grep -n "PDF page" for line numbers). Extract every retrievable equation, claim,
validity statement, assumption and empirical constant with verbatim quotes. Skip reference lists; do not emit a
claim per nomenclature line. One RESULT per window with target = "pages A-B".
`

phase('Extract')
// Small sequential batches: a session-limit hit then costs at most one batch, and every
// finished group is already journaled for resume.
const BATCH = args.batch || 3
const results = new Array(GROUP_NAMES.length).fill(null)
for (let b = 0; b < GROUP_NAMES.length; b += BATCH) {
  const idx = GROUP_NAMES.slice(b, b + BATCH).map((_, k) => b + k)
  const out = await parallel(idx.map(i => () =>
    agent(`${RULES}

YOUR GROUP: "${GROUP_NAMES[i]}". Read the JSON file ${GROUPS_PATH}; take the entry in "groups" whose name is "${GROUP_NAMES[i]}".
It lists model_targets (may be empty) and windows (may be empty). Process ALL of them. If you cannot finish one,
list it in "skipped" with the reason - never fabricate a result for it.`,
      { label: `extract:${GROUP_NAMES[i]}`, phase: 'Extract', schema: GROUP_SCHEMA })
  ))
  idx.forEach((i, k) => { results[i] = out[k] })
  log(`batch ${b / BATCH + 1}: ${out.filter(Boolean).length}/${idx.length} groups done`)
}

const done = results.map((r, i) => r ? { name: GROUP_NAMES[i], ...r } : null).filter(Boolean)
log(`groups done: ${done.length}/${GROUP_NAMES.length}`)
return { groups: done, failed: GROUP_NAMES.filter((_, i) => !results[i]) }
