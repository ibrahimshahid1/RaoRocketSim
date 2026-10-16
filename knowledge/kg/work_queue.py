"""Deterministic, resumable review queue. This module never starts an LLM."""
from __future__ import annotations

from collections import Counter
import json
import re

from . import store
from .index import tokenize
from .graph import model_source_support, equation_formulation_state

CHECKED = {'SUPPORTED', 'PARTIALLY_SUPPORTED', 'NOT_FOUND_IN_SOURCE', 'SOURCE_UNREADABLE'}


def build() -> dict:
    docs = store.index_by('manifest')
    models = store.load('models')
    tasks, blocked = [], []
    for m in models:
        d = docs.get(m.get('source_document_id'))
        effective, support_note = model_source_support(m, d)
        if effective in CHECKED and m.get('source_support_sha256') == (d or {}).get('sha256'):
            continue
        rec = {'target': m['id'], 'registry_id': m['registry_id'], 'subsystem': m['subsystem'],
               'doc_id': m.get('source_document_id'), 'equation_ref': m.get('equation_ref'),
               'relation': m.get('relation'), 'basis_class': m.get('basis_class'),
               'effective_source_support_state': effective, 'source_review_note': support_note,
               'prior_declared_support': m.get('source_support_state', 'NOT_CHECKED')}
        if d and d.get('markdown_exists') and d.get('page_count') != 0 and d.get('conversion', {}).get('status') != 'pdf_conversion_failed':
            rec.update({'state': 'PENDING', 'source_pdf': d['source_pdf'],
                        'markdown': d.get('generated_markdown'), 'source_sha256': d.get('sha256'),
                        'page_count': d.get('page_count')})
            tasks.append(rec)
        else:
            rec['state'] = ('SOURCE_UNREADABLE' if d and d.get('page_count') == 0 else
                            'REPOSITORY_BASIS_REVIEW' if m.get('source_kind') == 'repository_document' else 'NO_LOCAL_SOURCE_ROUTE')
            rec['local_source'] = m.get('local_source')
            blocked.append(rec)
    tasks.sort(key=lambda r: (r.get('doc_id') or '', r['target']))
    objects = [r for name in ('equations', 'claims') for r in store.load(name)]
    verify = [{'target': r['id'], 'doc_id': r.get('document_id'), 'page': r.get('page'),
               'model_ids': r.get('model_ids', []), 'state': 'PENDING_VERIFICATION'}
              for r in objects if r.get('verification') not in ('CONFIRMED', 'REFUTED')
              and r.get('active', True) and r.get('source_state', 'CURRENT') == 'CURRENT']
    verify.sort(key=lambda r: (not bool(r['model_ids']), r.get('doc_id') or '', r.get('page') or 0, r['target']))
    extracted_docs = {r.get('document_id') for name in ('equations', 'claims', 'entities', 'validity')
                      for r in store.load(name) if r.get('document_id') and r.get('active', True) and r.get('source_state', 'CURRENT') == 'CURRENT'}
    repairs = [{'target': e['id'], 'doc_id': e.get('document_id'), 'page': e.get('page'),
                'state': 'INCOMPLETE_TRANSCRIPTION'} for e in store.load('equations')
               if equation_formulation_state(e) == 'INCOMPLETE_TRANSCRIPTION'
               and e.get('active', True) and e.get('source_state', 'CURRENT') == 'CURRENT']
    return {'summary': {'models_total': len(models),
                        'model_support': dict(Counter(m.get('source_support_state', 'NOT_CHECKED') for m in models)),
                        'effective_model_support': dict(Counter(model_source_support(m, docs.get(m.get('source_document_id')))[0] for m in models)),
                        'pending_source_reviews': len(tasks), 'needs_source_routing': len(blocked),
                        'pending_verifications': len(verify),
                        'documents_with_extracted_objects': len(extracted_docs),
                        'canonical_documents_without_extracted_objects': sum(not d.get('alternate_copy_of') and d['id'] not in extracted_docs for d in docs.values())},
            'model_reviews': tasks, 'source_routing': blocked, 'verification': verify, 'formula_repairs': repairs,
            'policy': 'Process one source at a time; bank each completed result. No automatic agents or paid calls. A source review is not independent equation verification.'}


def write() -> dict:
    result = build()
    store.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (store.REPORT_DIR / 'extraction_queue.json').write_text(json.dumps(result, indent=1, ensure_ascii=False))
    return result


def packet(target: str, max_chars: int = 16000) -> dict:
    """Export a bounded reading packet; it is not an exhaustive source search."""
    if max_chars < 500:
        raise ValueError('max_chars must be at least 500')
    model = store.index_by('models').get(target)
    obj = model
    if obj is None:
        obj = next((r for name in ('equations', 'claims') for r in store.load(name) if r['id'] == target), None)
    if obj is None:
        return {'target': target, 'state': 'UNRESOLVED_ENTITY'}
    did = obj.get('source_document_id') if model else obj.get('document_id')
    doc = store.index_by('manifest').get(did)
    if not doc:
        return {'target': target, 'state': 'SOURCE_NOT_INGESTED', 'object': obj}
    conn = store.connect()
    chunks = store.doc_chunks(conn, did)
    conn.close()
    terms = set(tokenize(' '.join(str(obj.get(k, '')) for k in ('quantity', 'relation', 'equation_ref', 'label', 'statement'))))
    def relevance(chunk):
        if not model and chunk['page_start'] <= (obj.get('page') or 0) <= chunk['page_end']:
            return 10000
        return len(terms.intersection(tokenize(chunk['text'])))
    selected, consumed = [], 0
    for chunk in sorted(chunks, key=lambda c: (-relevance(c), c['seq'])):
        if consumed + len(chunk['text']) > max_chars:
            continue  # keep equations/tables atomic; never crop a source passage
        selected.append({k: chunk[k] for k in ('chunk_id', 'page_start', 'page_end', 'section', 'text')})
        consumed += len(chunk['text'])
    selected.sort(key=lambda c: c['chunk_id'])
    return {'target': target, 'state': 'OK' if selected else 'INSUFFICIENT_SOURCE_CONTEXT', 'object': obj,
            'document': {k: doc.get(k) for k in ('id', 'title', 'source_pdf', 'generated_markdown', 'sha256', 'page_count')},
            'passages': selected, 'source_chars': consumed, 'max_source_chars': max_chars,
            'complete_document': len(selected) == len(chunks) and bool(chunks),
            'instructions': [
                'Read the source around these candidate passages and inspect the original PDF for equations or damaged notation.',
                'Quote contiguous source text verbatim and retain PDF page numbers. The packet is not evidence of absence elsewhere.',
                'Do not mark NOT_FOUND_IN_SOURCE from a bounded packet alone. Leave unchecked work pending.',
                'Use workflows/extraction.schema.json and bank a single result before starting the next task.',
            ]}
