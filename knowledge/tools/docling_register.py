"""Register checkpointed Docling imports in the existing KG, preserving parser scope.

Run after docling_import.py. Does not infer verified claims or change model support.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'knowledge'))
sys.path.insert(0,str(Path(__file__).parent))
from kg.ingest import build_manifest
from kg.enrichment import integrate
from docling_enrich import publish_attempt, model_revisions
from sync_corpus import load_manifest, _write_manifest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bibliography-json',type=Path)
    ap.add_argument('--conversion-log',type=Path)
    args=ap.parse_args()
    os.environ.setdefault('HF_HOME',str(ROOT/'knowledge/.work/huggingface'))
    agents=ROOT/'propulsion_texts/propulsion_texts_for_agents'
    rows=load_manifest()
    bibliography=json.loads(args.bibliography_json.read_text())['documents'] if args.bibliography_json else []
    bib={r['sha256']:r for r in bibliography}
    warnings=set()
    if args.conversion_log:
        log=args.conversion_log.read_text(errors='replace')
        warnings={Path(p).name for p in re.findall(r'OSD failed \(doc (.*?), page:',log)}
        (agents/'docling_import'/'conversion.log').write_text(log)
    imported=[r for r in rows if r.get('converter')=='docling' and r.get('docling_artifacts')]
    index_path=agents/'paper_index.json'
    paper_index=json.loads(index_path.read_text())
    index_md=agents/'paper_index.md'
    index_lines=[]
    for r in imported:
        info=bib.get(r['sha256'],{})
        if info:
            r.update(pdf_title=info['title'],pdf_author=', '.join(info['authors']),source_identifier=info.get('doi',''))
        if r['source_pdf'] in warnings:
            note='Tesseract orientation detection warned on sparse OCR crops; Docling returned success on all requested pages. Equations and OCR still require page comparison. See docling_import/conversion.log.'
            if note not in r['issues']: r['issues'].append(note)
        if info and not any(x.get('PDF path')==r['source_pdf'] for x in paper_index):
            pid='PAPER-DOCLING-'+r['sha256'][:12].upper()
            paper_index.append({'Paper ID':pid,'Title':info['title'],'Authors':', '.join(info['authors']),
                'Year':info['year'],'DOI or report number':info.get('doi',''),
                'PDF path':r['source_pdf'],'Markdown path':r['output_markdown'],
                'Topics':['CFD','Nozzle design','Propulsion'],'Keywords':[],
                'Extraction status':r['extraction_status'],
                'Metadata basis':info.get('metadata_basis','Original PDF identity inventory')})
            index_lines.append(f'| {pid} | {info["title"]} | {", ".join(info["authors"])} | {info["year"]} | CFD, Nozzle design, Propulsion | [{r["source_pdf"]}](<../{r["source_pdf"]}>) | [{r["output_markdown"]}](<{r["output_markdown"]}>) | needs_review |')
        # Canonical Markdown uses portable paths; preserve checkpoint/raw artifacts.
        out=agents/r['output_markdown']
        dest=ROOT/r['docling_artifacts']
        prefix=Path(os.path.relpath(dest,out.parent)).as_posix()
        text=out.read_text().replace(str(dest/'images')+'/',prefix+'/images/')
        out.write_text(text)
        r['markdown_chars']=len(text)
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
            if target.startswith(('http:','https:','#','mailto:')): continue
            if not (out.parent/target).is_file(): raise RuntimeError('Broken Markdown asset: '+target)
        markers=list(map(int,re.findall(r'<!-- PDF page (\d+) -->',text)))
        if markers!=list(range(1,r['page_count']+1)): raise RuntimeError('Incorrect page markers: '+str(out))
    _write_manifest(rows)
    index_path.write_text(json.dumps(paper_index,indent=1,ensure_ascii=False)+'\n')
    if index_lines:
        with index_md.open('a') as stream: stream.write('\n'+'\n'.join(index_lines)+'\n')
    documents={d['sha256']:d for d in build_manifest()}
    reports=[]
    document_reports=[]
    for row in imported:
        d=documents[row['sha256']]
        dest=ROOT/row['docling_artifacts']
        for path in sorted(dest.glob('p*.meta.json')):
            batch=json.loads(path.read_text())
            raw=ROOT/batch['raw_artifact']
            metadata={k:batch[k] for k in ('source_pdf','source_sha256','page_range','options','versions','conversion_status','elapsed_seconds')}
            metadata.update(document_id=d['id'],model_revisions=model_revisions(),
                            model_revision_scope='Observed cached refs, not runtime attestation.',
                            observed_pages=list(range(batch['page_range'][0],batch['page_range'][1]+1)),
                            attempt_id='bulk-'+hashlib.sha256(path.read_bytes()).hexdigest(),
                            attempt_created_at=batch['converted_at'])
            md='\n\n'.join(f'<!-- PDF page {r["page"]} -->\n'+(ROOT/r['path']).read_text() for r in batch['pages'])
            md=re.sub(r'(?<=\()images/', '../images/', md)
            md=md.replace(str(dest/'images')+'/', '../images/')
            bank=dest/'kg'
            bank.mkdir(exist_ok=True)
            for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',md):
                if target.startswith(('http:','https:','#','mailto:')): continue
                if not (bank/target).is_file(): raise RuntimeError('Broken banked Markdown asset: '+target)
            raw_pub,meta_pub=publish_attempt(raw.read_bytes(),md.encode(),metadata,dest=bank,root=ROOT)
            reports.append(integrate(raw_pub,meta_pub))
        document_reports.append({'document_id':d['id'],'source_pdf':d['source_pdf'],
            'source_sha256':d['sha256'],'page_count':d['page_count'],
            'generated_markdown':d['generated_markdown'],
            'markdown_sha256':hashlib.sha256((ROOT/d['generated_markdown']).read_bytes()).hexdigest(),
            'figure_crops':row['docling_picture_count'],'tables':row['docling_table_count'],
            'page_markers_verified':True,'all_local_markdown_links_verified':True,
            'conversion_issues':row['issues'],'parser_artifacts':row['docling_artifacts']})
        print(json.dumps({'document_id':d['id'],'pages':d['page_count'],'figure_crops':row['docling_picture_count'],'state':'REGISTERED_DOC_AND_CANDIDATES'}),flush=True)
    report={'documents':len(imported),'pages':sum(r['page_count'] for r in imported),'figures':sum(r['docling_picture_count'] for r in imported),'document_reports':document_reports,'batches':reports,'scope':'Full-page local Docling layout/OCR/table import; unreviewed parser formulas; linked figure crops and every original page facsimile; no scientific support promotion.'}
    (ROOT/'knowledge/reports/moc_cfd_docling_import_2026-11-21.json').write_text(json.dumps(report,indent=1)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('batches','document_reports')}))

if __name__=='__main__': main()
