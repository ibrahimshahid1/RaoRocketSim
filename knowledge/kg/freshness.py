"""Preserve historical extraction records while invalidating changed evidence."""
from __future__ import annotations

from . import store


def snapshot() -> dict:
    states = {}
    if store.CHUNK_DB.exists():
        conn = store.connect()
        states = {r['doc_id']: dict(r) for r in conn.execute('SELECT * FROM doc_state')}
        conn.close()
    return {'documents': store.index_by('manifest'), 'chunks': states}


def reconcile(before: dict) -> dict:
    """A changed PDF or derivative requires a new review, even if IDs survived.

    Existing objects are retained with their old hashes. They must not continue
    to support current implementations just because a chunk ID was reused.
    """
    after = snapshot()
    changed = {}
    for did, old in before['documents'].items():
        new = after['documents'].get(did)
        old_chunk = before['chunks'].get(did, {})
        new_chunk = after['chunks'].get(did, {})
        if new is None:
            changed[did] = 'document removed from manifest'
        elif old.get('sha256') != new.get('sha256'):
            changed[did] = 'source PDF checksum changed'
        elif old_chunk and not new_chunk:
            changed[did] = 'source is no longer ingested'
        elif old_chunk and old_chunk.get('markdown_sha256') != new_chunk.get('markdown_sha256'):
            changed[did] = 'generated Markdown checksum changed'
    invalidated = 0
    for name in ('equations', 'claims', 'entities', 'validity'):
        records = store.load(name)
        dirty = False
        for rec in records:
            did = rec.get('document_id')
            if not did:
                continue
            old = before['documents'].get(did, {})
            if old.get('sha256') and not rec.get('source_sha256'):
                rec['source_sha256'] = old['sha256']
                rec['source_hash_basis'] = 'manifest at first freshness audit; original extraction did not record a hash'
                dirty = True
            reason = changed.get(did)
            current = after['documents'].get(did, {})
            if rec.get('source_sha256') and rec['source_sha256'] != current.get('sha256'):
                reason = reason or 'extraction checksum does not match current source'
            if reason:
                rec['source_state'] = 'STALE_SOURCE'
                rec['source_state_reason'] = reason
                invalidated += 1
                dirty = True
        if dirty:
            store.save(name, records, meta=store.load_meta(name))
    models = store.load('models')
    dirty = False
    for model in models:
        did = model.get('source_document_id')
        if did in changed and model.get('source_support_state') not in (None, 'NOT_CHECKED'):
            model.setdefault('source_support_history', []).append({
                'state': model.get('source_support_state'), 'notes': model.get('source_support_notes'),
                'source_sha256': before['documents'].get(did, {}).get('sha256'),
                'invalidated_because': changed[did],
            })
            model.pop('source_support_state_verified', None)
            model.pop('source_support_verifier_note', None)
            model['source_support_state'] = 'NOT_CHECKED'
            model['source_support_notes'] = 'Previous review is stale: ' + changed[did]
            dirty = True
    if dirty:
        store.save('models', models, meta=store.load_meta('models'))
    return {'changed_documents': changed, 'invalidated_objects': invalidated}
