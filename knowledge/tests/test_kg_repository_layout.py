"""Explicit relocation routes preserve evidence identities and review boundaries."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from kg import repository_layout as layout, store
from kg import implementation_freshness as freshness


def test_relocation_preserves_ids_science_and_historical_locators(tmp_path,monkeypatch):
    monkeypatch.setattr(store,'KNOWLEDGE_ROOT',tmp_path/'knowledge')
    mapping=tmp_path/layout.MAP_FILE; mapping.parent.mkdir(parents=True)
    mapping.write_text(json.dumps({'paths':{'package':{'raosim/old.py':'src/lrekit/components/new.py'}}}))
    source=tmp_path/'src/lrekit/components/new.py';source.parent.mkdir(parents=True)
    source.write_text('def solve():\n    return 1\n')
    store.save('software_links',[{'id':'SW_STABLE','type':'SoftwareComponent','file':'raosim/old.py',
        'qualname':'raosim.old:solve','line':20,'implementation_scientific_comparison_state':'PREVIOUS_REVIEW_RECORDED'}])
    store.save('models',[{'id':'MODEL_STABLE','source_support_state':'PARTIALLY_SUPPORTED','source_reviews':[{'file':'raosim/old.py'}]}])
    old_models=store.store_path('models').read_bytes()
    assert layout.reconcile_locators(root=tmp_path)['software_links']==1
    row,=store.load('software_links')
    assert row['id']=='SW_STABLE' and row['file']=='src/lrekit/components/new.py'
    assert row['repository_locator_history'][0]['file']=='raosim/old.py'
    assert row['implementation_scientific_comparison_state']=='NOT_RECHECKED'
    assert row['implementation_scientific_comparison_history'][0]['state']=='PREVIOUS_REVIEW_RECORDED'
    result=freshness.reconcile(repo_root=tmp_path)
    assert result['verified']==1
    assert store.load('software_links')[0]['line']==1
    assert store.store_path('models').read_bytes()==old_models
    assert layout.reconcile_locators(root=tmp_path)['software_links']==0


def test_checked_in_layout_resolves_all_stages_and_current_calls():
    root=Path(__file__).resolve().parents[2]
    current=layout.Layout(root)
    file=current.path('raosim/moc_bounded.py')
    assert file=='src/lrekit/components/nozzle/solvers/bounded.py'
    assert (root/file).is_file()
    assert current.path('src/raosim/components/nozzle/solvers/bounded.py')==file
    assert current.module('raosim.mdo.nlp._make_callables.solve')=='lrekit.optimization.nlp._make_callables.solve'
    assert current.module_for_file(root/file)=='lrekit.components.nozzle.solvers.bounded'
