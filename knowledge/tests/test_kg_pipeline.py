"""Corpus-free regression tests for preservation, sync and incremental rebuilds."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kg import store, ingest, freshness, graph, ground, cli, pipeline
from kg.chunker import chunk_markdown
from tools import sync_corpus


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(store, 'KNOWLEDGE_ROOT', tmp_path / 'knowledge')
    monkeypatch.setattr(store, 'CHUNK_DB', tmp_path / 'index' / 'chunks.sqlite')
    monkeypatch.setattr(graph, 'GRAPH_DIR', tmp_path / 'graph')
    monkeypatch.setattr(ingest, 'REPO_ROOT', tmp_path)
    return tmp_path


def document(root, sha='abc'):
    md = root / 'paper.md'
    md.write_text('<!-- PDF page 1 -->\n\n# Example\n\nPressure increases with temperature under fixed volume. This is a synthetic testing example.\n')
    return {'id': 'DOC_TEST', 'sha256': sha, 'markdown_exists': True, 'generated_markdown': 'paper.md',
            'conversion': {'status': 'good'}, 'filename': 'paper.pdf', 'source_pdf': 'paper.pdf'}


def test_grounding_rejects_reordered_words_and_changed_meaning(isolated):
    store.save('manifest', [document(isolated)])
    ingest.ingest_chunks(verbose=False)
    g = ground.DocGrounder('DOC_TEST')
    assert g.locate('Pressure increases with temperature under fixed volume.', 1)['state'] == 'GROUNDED'
    assert g.locate('Temperature increases with pressure under fixed volume.', 1)['state'] == 'UNGROUNDED'
    assert g.locate('Pressure decreases with temperature under fixed volume.', 1)['state'] == 'UNGROUNDED'


def test_legacy_scanned_manifest_is_low_confidence_even_if_old_flag_is_false(isolated, monkeypatch):
    corpus = isolated / 'propulsion_texts'
    agents = corpus / 'propulsion_texts_for_agents'
    agents.mkdir(parents=True)
    monkeypatch.setattr(ingest, 'CORPUS_ROOT', corpus)
    monkeypatch.setattr(ingest, 'AGENT_CORPUS_ROOT', agents)
    (agents / 'paper.md').write_text('<!-- PDF page 1 -->\n\nLegacy OCR passage requiring source verification. This longer synthetic paragraph exercises the readable chunk path while retaining the OCR confidence flag.')
    (agents / '_conversion_manifest.json').write_text(json.dumps([{
        'source_pdf': 'paper.pdf', 'output_markdown': 'paper.md', 'sha256': 'original',
        'extraction_status': 'needs_review', 'scanned': True, 'low_confidence': False,
    }]))
    docs = ingest.build_manifest(verbose=False)
    assert docs[0]['conversion']['low_confidence'] is True
    assert ingest.ingest_chunks(verbose=False)['states'][0]['state'] == 'LOW_EXTRACTION_CONFIDENCE'


def test_incremental_ingest_records_unchanged_and_removes_missing_derivative(isolated):
    d = document(isolated)
    store.save('manifest', [d])
    assert ingest.ingest_chunks(verbose=False)['ingested'] == 1
    result = ingest.ingest_chunks(verbose=False)
    assert result['unchanged'] == 1
    assert result['states'][0]['state'] == 'INGESTED'
    assert len(store.load('ingestion_state')) == 1
    (isolated / 'paper.md').unlink()
    result = ingest.ingest_chunks(verbose=False)
    assert result['removed'] == 1
    conn = store.connect()
    assert conn.execute('SELECT COUNT(*) FROM chunks').fetchone()[0] == 0
    conn.close()


def test_same_text_new_pdf_hash_reingests_and_invalidates_review(isolated):
    d = document(isolated)
    store.save('manifest', [d])
    ingest.ingest_chunks(verbose=False)
    store.save('equations', [{'id': 'EQ_TEST', 'document_id': d['id'], 'quote': 'Pressure increases'}])
    store.save('models', [{'id': 'MODEL_TEST', 'source_document_id': d['id'], 'source_support_state': 'SUPPORTED'}])
    before = freshness.snapshot()
    d['sha256'] = 'replacement'
    store.save('manifest', [d])
    assert ingest.ingest_chunks(verbose=False)['ingested'] == 1
    result = freshness.reconcile(before)
    assert result['invalidated_objects'] == 1
    eq = store.load('equations')[0]
    assert eq['source_sha256'] == 'abc'
    assert eq['source_state'] == 'STALE_SOURCE'
    assert store.load('models')[0]['source_support_state'] == 'NOT_CHECKED'
    assert store.load('models')[0]['source_support_history'][0]['state'] == 'SUPPORTED'


def test_changed_markdown_invalidates_without_deleting_historical_records(isolated):
    d = document(isolated)
    store.save('manifest', [d])
    ingest.ingest_chunks(verbose=False)
    store.save('claims', [{'id': 'CLM_TEST', 'document_id': d['id'], 'source_sha256': 'abc'}])
    before = freshness.snapshot()
    (isolated / 'paper.md').write_text('<!-- PDF page 1 -->\n\nCorrected transcription differs from the original.\n')
    ingest.ingest_chunks(verbose=False)
    assert 'Markdown' in freshness.reconcile(before)['changed_documents'][d['id']]
    assert store.load('claims')[0]['source_state'] == 'STALE_SOURCE'


def test_sync_checks_output_existence_and_conversion_failure(tmp_path, monkeypatch):
    corpus = tmp_path / 'propulsion_texts'
    agents = corpus / 'propulsion_texts_for_agents'
    markdown = agents / 'markdown'
    markdown.mkdir(parents=True)
    for name, value in [('REPO', tmp_path), ('CORPUS', corpus), ('AGENTS', agents),
                        ('MARKDOWN', markdown), ('MANIFEST', agents / '_conversion_manifest.json')]:
        monkeypatch.setattr(sync_corpus, name, value)
    pdf = corpus / 'paper.pdf'
    pdf.write_bytes(b'local test payload')
    row = {'source_pdf': 'paper.pdf', 'sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(),
           'output_markdown': 'markdown/paper.md', 'extraction_status': 'good'}
    sync_corpus.MANIFEST.write_text(json.dumps([row]))
    assert len(sync_corpus.scan()['missing_markdown']) == 1
    (markdown / 'paper.md').write_text('A derivative exists.')
    assert len(sync_corpus.scan()['synchronized']) == 1
    row['extraction_status'] = 'pdf_conversion_failed'
    sync_corpus.MANIFEST.write_text(json.dumps([row]))
    assert len(sync_corpus.scan()['missing_markdown']) == 1


def test_sync_does_not_call_missing_archive_synchronized(tmp_path, monkeypatch):
    corpus = tmp_path / 'propulsion_texts'
    agents = corpus / 'propulsion_texts_for_agents'
    markdown = agents / 'markdown'
    markdown.mkdir(parents=True)
    for name, value in [('REPO', tmp_path), ('CORPUS', corpus), ('AGENTS', agents),
                        ('MARKDOWN', markdown), ('MANIFEST', agents / '_conversion_manifest.json')]:
        monkeypatch.setattr(sync_corpus, name, value)
    (markdown / 'paper.md').write_text('An archive-derived passage.')
    row = {'source_pdf': 'archive/paper.pdf', 'sha256': 'original',
           'output_markdown': 'markdown/paper.md', 'source_kind': 'zip_member',
           'source_container': 'archive.zip', 'archive_member': 'paper.pdf'}
    sync_corpus.MANIFEST.write_text(json.dumps([row]))
    result = sync_corpus.scan()
    assert result['synchronized'] == []
    assert len(result['stale_markdown']) == 1
    assert 'archive is unavailable' in result['stale_markdown'][0]['reason']
    payload = b'original source payload'
    row['sha256'] = hashlib.sha256(payload).hexdigest()
    sync_corpus.MANIFEST.write_text(json.dumps([row]))
    with zipfile.ZipFile(corpus / 'archive.zip', 'w') as archive:
        archive.writestr('paper.pdf', payload)
    assert len(sync_corpus.scan()['synchronized']) == 1
    with zipfile.ZipFile(corpus / 'archive.zip', 'w') as archive:
        archive.writestr('paper.pdf', b'replaced source payload')
    result = sync_corpus.scan()
    assert result['synchronized'] == []
    assert 'checksum differs' in result['stale_markdown'][0]['reason']


def test_sync_classifies_exact_catalog_mirror_without_inventing_pdf(tmp_path, monkeypatch):
    corpus = tmp_path / 'propulsion_texts'
    agents = corpus / 'propulsion_texts_for_agents'
    markdown = agents / 'markdown'
    (markdown / 'pintle_injector').mkdir(parents=True)
    (corpus / 'pintle_injector').mkdir()
    for name, value in [('REPO', tmp_path), ('CORPUS', corpus), ('AGENTS', agents),
                        ('MARKDOWN', markdown), ('MANIFEST', agents / '_conversion_manifest.json')]:
        monkeypatch.setattr(sync_corpus, name, value)
    note = corpus / 'pintle_injector/PINTLE_LITERATURE_CATALOG.md'
    mirror = markdown / 'pintle_injector/PINTLE_LITERATURE_CATALOG.md'
    note.write_text('Human-authored bibliographic catalog, not a paper.')
    mirror.write_bytes(note.read_bytes())
    result = sync_corpus.scan()
    assert len(result['mirrored_reference_notes']) == 1
    assert not result['orphan_markdown']
    assert result['pdf_count'] == 0
    mirror.write_text('Changed content without a source mapping.')
    assert len(sync_corpus.scan()['orphan_markdown']) == 1


def test_cli_rebuild_uses_complete_pipeline(monkeypatch):
    args = []
    monkeypatch.setattr(pipeline, 'main', lambda argv: args.extend(argv) or 7)
    assert cli.main(['rebuild', '--force-chunks', '--skip-sync']) == 7
    assert args == ['--force-chunks', '--skip-sync']


def test_graph_keeps_stale_node_but_not_its_asserted_support(isolated):
    store.save('equations', [{'id': 'EQ_STALE', 'document_id': 'DOC_TEST', 'model_ids': ['MODEL_TEST'],
                              'source_state': 'STALE_SOURCE', 'form': 'old'}])
    result = graph.assemble()
    assert any(n['id'] == 'EQ_STALE' for n in result['nodes'])
    assert not any(e['from'] == 'EQ_STALE' or e['to'] == 'EQ_STALE' for e in result['edges'])


def test_queue_skips_completed_reviews_and_lists_missing_source_separately(isolated):
    from kg import work_queue
    d = document(isolated)
    store.save('manifest', [d])
    store.save('equations', [{'id': 'EQ_OLD', 'document_id': d['id'], 'form': 'NOT RENDERED',
                              'source_state': 'SUPERSEDED_EXTRACTION', 'active': False}])
    store.save('models', [
        {'id': 'MODEL_DONE', 'registry_id': 'done', 'subsystem': 'nozzle',
         'source_document_id': d['id'], 'source_support_state': 'PARTIALLY_SUPPORTED', 'source_support_sha256': 'abc'},
        {'id': 'MODEL_NEXT', 'registry_id': 'next', 'subsystem': 'nozzle', 'source_document_id': d['id']},
        {'id': 'MODEL_MISSING', 'registry_id': 'missing', 'subsystem': 'nozzle'},
    ])
    result = work_queue.build()
    assert [r['target'] for r in result['model_reviews']] == ['MODEL_NEXT']
    assert [r['target'] for r in result['source_routing']] == ['MODEL_MISSING']
    assert result['summary']['models_total'] == 3
    assert result['formula_repairs'] == []


def test_packet_is_bounded_and_does_not_claim_exhaustive_coverage(isolated):
    from kg import work_queue
    d = document(isolated)
    (isolated / 'paper.md').write_text('\n\n'.join(
        f'<!-- PDF page {i} -->\n\n## Section {i}\n\n' + ('Pressure temperature synthetic fixture. ' * 30)
        for i in range(1, 5)))
    store.save('manifest', [d])
    store.save('models', [{'id': 'MODEL_TEST', 'source_document_id': d['id'], 'relation': 'Pressure temperature'}])
    ingest.ingest_chunks(verbose=False)
    result = work_queue.packet('MODEL_TEST', max_chars=1400)
    assert 0 < result['source_chars'] <= 1400
    assert result['complete_document'] is False
    assert result['document']['sha256'] == 'abc'
    assert result['passages'][0]['page_start'] >= 1
    assert work_queue.packet('UNKNOWN')['state'] == 'UNRESOLVED_ENTITY'
