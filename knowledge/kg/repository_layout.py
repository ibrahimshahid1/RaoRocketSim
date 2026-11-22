"""Translate current code locators through the explicit repository relocation map.

Stable object IDs and scientific source reviews are deliberately not renamed.
Translation establishes a path route only; AST freshness supplies locator checks.
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
from . import REPO_ROOT, store

MAP_FILE='docs/architecture/relocations_2026-11-21.json'

class Layout:
    def __init__(self, root=REPO_ROOT):
        self.root=Path(root)
        path=self.root/MAP_FILE
        payload=json.loads(path.read_text()) if path.is_file() else {}
        self.paths={k:v for group in payload.get('paths',{}).values() for k,v in group.items()}
        self.prefixes=sorted(self.paths,key=len,reverse=True)
        self.modules={self._module(k):self._module(v) for k,v in self.paths.items()
                      if k.startswith(('raosim/','lrekit/')) or k in ('raosim','lrekit')}
        self.modules.update(payload.get('module_aliases',{}).get('map',{}))
        self.module_prefixes=sorted(self.modules,key=len,reverse=True)

    @staticmethod
    def _module(path):
        path=path.removeprefix('src/').removesuffix('.py').removesuffix('/__init__')
        return path.replace('/','.')

    def path(self, value):
        value=str(value)
        # Resolve the intermediate src/raosim layout recorded between both moves.
        if self.paths and value.startswith('src/raosim/'):
            return 'src/lrekit/'+value[len('src/raosim/'):]
        for old in self.prefixes:
            if value==old or value.startswith(old+'/'):
                return self.paths[old]+value[len(old):]
        return value

    def module(self, value):
        for old in self.module_prefixes:
            if value==old or value.startswith(old+'.'):
                return self.modules[old]+value[len(old):]
        if self.paths and value.startswith('raosim.'):
            return 'lrekit.'+value[len('raosim.'):]
        return value

    def qualname(self,value):
        if ':' in value:
            module,symbol=value.split(':',1)
            return (self.path(module) if '/' in module else self.module(module))+':'+symbol
        return self.module(value)

    def module_for_file(self,path):
        return self._module(str(Path(path).relative_to(self.root)))

    def code_roots(self):
        if (self.root/'src/lrekit').is_dir():
            return [self.root/'src/lrekit',self.root/'src/raosim',self.root/'scripts',self.root/'tests/experiments']
        return [self.root/'raosim',self.root/'lrekit',self.root/'scripts']


def reconcile_locators(*,root=REPO_ROOT):
    """Preserve IDs/history; translate active code/test locator fields only."""
    layout=Layout(root)
    counts={}
    for name in ('software_links','tests'):
        rows=store.load(name);count=0
        for row in rows:
            if name=='software_links' and row.get('type') not in ('SoftwareComponent','EmpiricalConstant'):
                continue
            changes={}
            for key in ('file','qualname','path'):
                value=row.get(key)
                if not isinstance(value,str):continue
                if key=='qualname': translated=layout.qualname(value)
                elif key=='path':
                    file,*rest=value.split('::')
                    translated='::'.join([layout.path(file),*rest])
                else: translated=layout.path(value)
                if translated!=value:changes[key]=translated
            if not changes:continue
            row.setdefault('repository_locator_history',[]).append({
                **{k:row[k] for k in set(changes)|{'line','end_line','source_sha256','definition_sha256','implementation_scientific_comparison_state'} if k in row},
                'recorded_at':datetime.now(timezone.utc).isoformat(),
                'relocation_map':MAP_FILE,'scope':'Path translation only; scientific review not reconfirmed.'})
            row.update(changes,repository_locator_translation_state='EXPLICIT_RELOCATION_MAP',
                       repository_locator_translation_scope='Preserved object ID and historical scientific assertions; current AST and scientific comparisons checked separately.')
            if row.get('type')=='SoftwareComponent':
                if row.get('implementation_scientific_comparison_state') not in (None,'NOT_RECHECKED'):
                    row.setdefault('implementation_scientific_comparison_history',[]).append({
                        'state':row['implementation_scientific_comparison_state'],
                        'reason':'Explicit repository relocation; current implementation science not rechecked.',
                        'recorded_at':datetime.now(timezone.utc).isoformat()})
                row['implementation_scientific_comparison_state']='NOT_RECHECKED'
            count+=1
        if count:store.save(name,rows)
        counts[name]=count
    return counts
