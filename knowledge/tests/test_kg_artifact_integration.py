"""Artifact retrieval preserves scope and never revives historical binding."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kg import api, cli, graph, store


def test_artifacts_expose_provenance_without_promoting_reference_cad(tmp_path, monkeypatch):
    monkeypatch.setattr(store, 'KNOWLEDGE_ROOT', tmp_path)
    monkeypatch.setattr(graph, 'GRAPH_DIR', tmp_path / 'graph')
    records = [
        {'id': 'CFG_CURRENT', 'type': 'EngineConfiguration', 'name': 'Injector input snapshot',
         'scope': 'injector_layout_only', 'active': True, 'provenance_id': 'PROV_CFG',
         'producer_binding_state': 'UNKNOWN', 'qualification_state': 'NOT_QUALIFIED'},
        {'id': 'CAD_CURRENT', 'type': 'CADArtifact', 'name': 'Injector faceplate',
         'state': 'REPORT_ONLY', 'active': True, 'provenance_id': 'PROV_CAD',
         'observed_sha256': 'observed', 'generation_time_integrity_state': 'UNKNOWN',
         'binding_state': 'UNKNOWN', 'qualification_state': 'NOT_QUALIFIED'},
        {'id': 'CAD_OLD', 'type': 'CADArtifact', 'name': 'Injector old faceplate',
         'active': False, 'source_state': 'STALE_ARTIFACT_BYTES'},
        {'id': 'LINK_CURRENT', 'type': 'ArtifactLink', 'from': 'CAD_CURRENT', 'to': 'CFG_CURRENT',
         'edge': 'BOUND_TO_CONFIGURATION', 'provenance_id': 'PROV_LINK', 'active': True,
         'binding_state': 'UNKNOWN', 'binding_basis': 'same_report_layout_snapshot'},
        # Even an inconsistent old active link cannot revive an inactive node.
        {'id': 'LINK_OLD', 'type': 'ArtifactLink', 'from': 'CAD_OLD', 'to': 'CFG_CURRENT',
         'edge': 'BOUND_TO_CONFIGURATION', 'provenance_id': 'PROV_OLD', 'active': True},
    ]
    store.save('artifacts', records)
    store.save('provenance', [{'id': 'PROV_CAD', 'kind': 'PROJECT_DERIVES',
                               'statement': 'Existing report and current bytes only'}])
    assembled = graph.assemble()
    assert assembled['meta']['n_dangling_dropped'] == 0
    assert any(e['from'] == 'CAD_CURRENT' and e['binding_state'] == 'UNKNOWN'
               for e in assembled['edges'])
    assert not any(e['from'] == 'CAD_OLD' for e in assembled['edges'])
    api.reset_caches()
    found = api.find_artifacts('injector')
    assert {r['id'] for r in found['records']} == {'CFG_CURRENT', 'CAD_CURRENT'}
    detail = api.get_artifact('CAD_CURRENT')
    assert detail['provenance']['kind'] == 'PROJECT_DERIVES'
    assert detail['record']['generation_time_integrity_state'] == 'UNKNOWN'
    assert cli._sources('CAD_CURRENT') == detail
    assert api.get_artifact('CAD_OLD')['record']['active'] is False
    assert api.find_artifacts('missing')['state'] == 'NO_RELEVANT_SOURCE_FOUND'
    api.reset_caches()
