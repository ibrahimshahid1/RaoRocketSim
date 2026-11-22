"""Import source PDFs using local Docling, with page-anchored Markdown and images.

Checkpoint each bounded page batch. Source PDFs remain authoritative; layout,
OCR, formula text and figure boundaries are unreviewed parser output. Formula
model enrichment is deliberately disabled for bulk ingestion, not called verified.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'knowledge'))
sys.path.insert(0, str(Path(__file__).parent))
from kg.locks import heavy_job

AGENTS = ROOT / 'propulsion_texts/propulsion_texts_for_agents'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False) + '\n')
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('pdfs', nargs='+', type=Path)
    ap.add_argument('--batch-pages', type=int, default=16)
    ap.add_argument('--threads', type=int, default=6)
    args = ap.parse_args()
    os.environ.setdefault('HF_HOME', str(ROOT / 'knowledge/.work/huggingface'))
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    os.environ.setdefault('OMP_NUM_THREADS', str(args.threads))
    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import ImageRefMode
    from sync_corpus import load_manifest, _write_manifest
    import pypdfium2 as pdfium
    options = PdfPipelineOptions()
    options.do_formula_enrichment = False
    options.do_code_enrichment = False
    options.enable_remote_services = False
    options.allow_external_plugins = False
    options.do_table_structure = True
    options.ocr_options = TesseractCliOcrOptions(lang=['eng'])
    options.generate_picture_images = True
    options.generate_page_images = True
    options.images_scale = 1.5
    options.accelerator_options = AcceleratorOptions(num_threads=args.threads, device='cpu')
    options.document_timeout = 1800
    versions = {k: importlib.metadata.version(k) for k in ('docling', 'docling-core', 'docling-ibm-models', 'docling-parse', 'torch')}
    options_json = options.model_dump(mode='json')
    with heavy_job(purpose='docling_import source PDFs with linked figures'):
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
        summaries = []
        for path in args.pdfs:
            path = path.resolve()
            rel_pdf = path.relative_to(ROOT / 'propulsion_texts')
            sha = digest(path)
            pdf = pdfium.PdfDocument(path)
            page_count = len(pdf)
            pdf.close()
            dest = AGENTS / 'docling_import' / sha[:16]
            dest.mkdir(parents=True, exist_ok=True)
            assets = dest / 'images'
            assets.mkdir(exist_ok=True)
            config = {'source_sha256': sha, 'source_pdf': str(path.relative_to(ROOT)), 'versions': versions, 'options': options_json}
            config_path = dest / 'config.json'
            if config_path.exists() and json.loads(config_path.read_text()) != config:
                raise RuntimeError('Existing batch config differs; retain history and use a new destination: '+str(dest))
            atomic_json(config_path, config)
            batches = []
            for first in range(1, page_count+1, args.batch_pages):
                last = min(page_count, first + args.batch_pages - 1)
                meta_path = dest / f'p{first:04}-{last:04}.meta.json'
                raw_path = meta_path.with_name(meta_path.name.replace('.meta.json','.json'))
                if meta_path.exists():
                    batch = json.loads(meta_path.read_text())
                    if digest(raw_path) != batch['raw_sha256'] or any(digest(ROOT / r['path']) != r['sha256'] for r in batch['pages']):
                        raise RuntimeError('Checkpoint checksum differs: '+str(meta_path))
                    print(json.dumps({'state':'CHECKPOINT_REUSED', 'pdf':path.name,'pages':[first,last]}),flush=True)
                    batches.append(batch)
                    continue
                started = time.monotonic()
                result = converter.convert(path, page_range=(first,last))
                if result.status.value != 'success':
                    raise RuntimeError(f'Docling incomplete {path.name} {first}:{last}: {result.status.value} {result.errors}')
                document = result.document
                document.save_as_json(raw_path, image_mode=ImageRefMode.REFERENCED, artifacts_dir=assets)
                records = []
                for page in range(first,last+1):
                    if page not in document.pages:
                        raise RuntimeError(f'Docling omitted page {page}')
                    md_path = dest / f'page-{page:04}.md'
                    document.save_as_markdown(md_path, artifacts_dir=assets, image_mode=ImageRefMode.REFERENCED, page_no=page)
                    # Page images cover diagrams/equations that layout may miss.
                    image_path = assets / f'page-{page:04}.png'
                    document.pages[page].image.pil_image.save(image_path)
                    with md_path.open('a') as stream:
                        stream.write(f'\n\n[Original page {page} facsimile](images/{image_path.name})\n')
                    records.append({'page':page,'path':str(md_path.relative_to(ROOT)),'sha256':digest(md_path),'facsimile':str(image_path.relative_to(ROOT))})
                batch = {**config,'page_range':[first,last], 'conversion_status':result.status.value,'raw_artifact':str(raw_path.relative_to(ROOT)), 'raw_sha256':digest(raw_path), 'pages':records,'picture_count':len(document.pictures),'table_count':len(document.tables),'elapsed_seconds':round(time.monotonic()-started,2),'converted_at':datetime.now(timezone.utc).isoformat()}
                atomic_json(meta_path,batch)
                batches.append(batch)
                print(json.dumps({'state':'BATCH_SAVED','pdf':path.name,'pages':[first,last],'seconds':batch['elapsed_seconds'],'pictures':batch['picture_count']}),flush=True)
            out = AGENTS / 'markdown' / rel_pdf.with_suffix('.md')
            out.parent.mkdir(parents=True, exist_ok=True)
            now = datetime.now(timezone.utc).isoformat()
            header = '> [!WARNING]\n> Unreviewed Docling layout/OCR/table derivative. Formula enrichment is disabled; inspect equations, figures and numerical values in the linked original pages.\n\n'
            header += f'<!-- lrekit-kg-conversion\nsource_pdf: {path}\nsource_sha256: {sha}\nconverter: docling {versions["docling"]} (layout OCR tables referenced figures)\nconverted_at: {now}\nextraction_status: needs_review\n-->\n\n'
            body = []
            for batch in batches:
                for record in batch['pages']:
                    text = (ROOT / record['path']).read_text()
                    # Per-page artifacts and canonical Markdown have different roots.
                    prefix = Path(os.path.relpath(dest, out.parent)).as_posix()
                    text = re.sub(r'(?<=\()images/', prefix+'/images/',text)
                    text = text.replace(str(assets)+'/', prefix+'/images/')
                    body.append(f'<!-- PDF page {record["page"]} -->\n\n'+text)
            out.write_text(header+'\n\n'.join(body))
            row = {'source_pdf':str(rel_pdf),'output_markdown':str(out.relative_to(AGENTS)),'source_container':path.name,'source_kind':'file','sha256':sha,'source_bytes':path.stat().st_size,'page_count':page_count,'markdown_chars':out.stat().st_size,'extraction_status':'needs_review','low_confidence':True,'converter':'docling','converter_version':versions['docling'],'converted_at':now,'method':'docling_layout_ocr_tables_referenced_figures','issues':['Bulk Docling parser output is unreviewed; formula model enrichment disabled. All original pages linked as facsimiles.'],'docling_artifacts':str(dest.relative_to(ROOT)),'docling_picture_count':sum(b['picture_count'] for b in batches),'docling_table_count':sum(b['table_count'] for b in batches),'pdf_title':'','pdf_author':''}
            manifest=load_manifest()
            prior = [r for r in manifest if r.get('source_pdf')==str(rel_pdf)]
            if prior and prior[0].get('converter')!='docling':
                raise RuntimeError('Refusing to replace prior non-Docling canonical derivative')
            _write_manifest([r for r in manifest if r.get('source_pdf')!=str(rel_pdf)]+[row])
            summaries.append(row)
            atomic_json(AGENTS/'docling_import'/'latest_import.json',summaries)
            print(json.dumps({'state':'DOCUMENT_REGISTERED','pdf':path.name,'pages':page_count,'pictures':row['docling_picture_count']}),flush=True)

if __name__=='__main__':
    main()
