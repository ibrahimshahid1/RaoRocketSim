// Adversarial verification of grounded, model-linked equations (one skeptic per subsystem group).
// Launch with: Workflow({script: <this file>, args: {path: "knowledge/.work/verify_groups.json", groups: [...]}})
export const meta = {
  name: 'lrekit-extraction-verify',
  description: 'Skeptics re-read the source chunk for each model-linked equation/claim and try to refute it',
  phases: [{ title: 'Verify', detail: 'one skeptic agent per subsystem group' }],
}
const PATH = args.path
const GROUPS = args.groups
const ADD = { type: 'object', properties: { statement: { type: 'string' }, page: { type: 'integer' }, quote: { type: 'string' } }, required: ['statement', 'page', 'quote'] }
const VERDICT = {
  type: 'object',
  properties: {
    object_id: { type: 'string' },
    refuted: { type: 'boolean' },
    reason: { type: 'string' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
    missing_assumptions: { type: 'array', items: ADD },
    missing_validity: { type: 'array', items: ADD },
    model_support_override: { type: 'string', enum: ['', 'SUPPORTED', 'PARTIALLY_SUPPORTED', 'NOT_FOUND_IN_SOURCE'] },
  },
  required: ['object_id', 'refuted', 'reason', 'confidence'],
}
const SCHEMA = { type: 'object', properties: { group: { type: 'string' }, verdicts: { type: 'array', items: VERDICT }, notes: { type: 'string' } }, required: ['group', 'verdicts'] }
const RULES = `
You are a SKEPTIC for the LREKit knowledge system. You receive extracted equations/claims, each with the document,
page, verbatim anchor quote, the transcribed form, assumptions, validity and (for model-linked ones) the registry
relation it is claimed to support. Your job is to REFUTE where possible.
For each object: open the Markdown at the given path, Grep the anchor quote, read the surrounding page(s) (and the
PDF page rendered with gs if the Markdown is garbled: gs -sDEVICE=png16m -r150 -dFirstPage=N -dLastPage=N -o /tmp/p.png <pdf>).
Refute (refuted=true) if ANY of: the source does not state this form; the transcription changes the meaning
(dropped term, wrong exponent, wrong variable); the equation is presented under assumptions the extraction omitted
in a way that changes its applicability; the claim misattributes; the page is wrong by more than 2.
Do NOT refute for notation/OCR rendering differences that preserve meaning, or for missing information that the
source itself does not give. If uncertain, refuted=false with confidence=low and explain.
When the extractor dropped an assumption/validity limit the source states, add it under missing_assumptions /
missing_validity WITH a verbatim 60-240 char quote and page (it will be grounded automatically).
For model-linked objects, also judge the registry support state; set model_support_override only if you disagree
with the recorded state, else leave it empty.`
phase('Verify')
const out = await parallel(GROUPS.map(name => () =>
  agent(`${RULES}\n\nYOUR GROUP: "${name}". Read ${PATH}, take groups["${name}"] (a list of objects). Return one verdict per object_id.`,
    { label: `verify:${name}`, phase: 'Verify', schema: SCHEMA })))
return { groups: out.map((r, i) => r ? { name: GROUPS[i], ...r } : null).filter(Boolean), failed: GROUPS.filter((_, i) => !out[i]) }
