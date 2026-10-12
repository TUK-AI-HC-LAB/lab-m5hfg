"""Targeted, visually checked PDF cells vs local evidence; does not rewrite PDF."""
import csv,hashlib,json
from pathlib import Path
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parent
PDF=Path('C:/Users/test/Downloads/MuSc.pdf')
OUT=ROOT/'results/user_pdf_evidence_audit_20261012';OUT.mkdir(exist_ok=True)
RUN=ROOT/'results/tables3_7_seed42_20261012'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
means={(int(r['table']),r['dataset'],r['condition']):r for r in read(RUN/'mean_metrics.csv')}
bench={r['s']:r for r in read(RUN/'benchmark_summary.csv')}
assert json.loads((RUN/'verification.json').read_text())['status']=='passed'
rows=[]
def add(page,item,pdf_value,value,source):
    expected=f'{value:.1f}' if value is not None else ''
    rows.append(dict(pdf_page=page,item=item,pdf_displayed_value=pdf_value,latest_local_displayed_value=expected,
        status='local_not_completed' if value is None else 'matches_latest_local_rounded' if float(pdf_value)==float(expected) else 'does_not_match_latest_local_rounded',
        local_evidence=source))
def metric(page,table,ds,condition,key,pdf_value):
    add(page,f'Table{table}/{ds}/{condition}/{key}',pdf_value,100*float(means[table,ds,condition][key]),str(RUN/'mean_metrics.csv'))
metric(8,3,'mvtec_ad','5','image_auroc','95.4')
metric(8,3,'mvtec_ad','1,3,5','pixel_auroc','97.3')
metric(8,3,'visa','1,3,5','image_auroc','92.8')
metric(8,5,'mvtec_ad','w','image_f1_max','97.5')
metric(8,5,'visa','w','image_auroc','92.8')
metric(8,5,'visa','w','image_f1_max','89.5')
add(8,'Table2/IGD/image_auroc','93.4',None,'method14/markdown/IGD_mvtec_execution.md')
add(8,'Table2/IGD/pixel_auroc','93.0',None,'method14/markdown/IGD_mvtec_execution.md')
for s,t,m in [('1','726.5','7418.2'),('2','420.1','7070.7'),('3','327.7','6804.2')]:
    add(9,f'Table6/s{s}/core_ms_per_image',t,float(bench[s]['core_ms_per_image']),str(RUN/'benchmark_summary.csv'))
    add(9,f'Table6/s{s}/gpu_cost_MB',m,float(bench[s]['gpu_cost_MB']),str(RUN/'benchmark_summary.csv'))
for ds,s,ac,px in [('mvtec_ad','s1','97.8','97.1'),('mvtec_ad','s2','97.6','97.1'),('mvtec_ad','s3','97.4','97.1'),('visa','s1','92.5','98.7'),('visa','s2','92.5','98.7'),('visa','s3','92.4','98.7')]:
    metric(9,7,ds,s,'image_auroc',ac);metric(9,7,ds,s,'pixel_auroc',px)
with (OUT/'selected_pdf_cells.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
reader=PdfReader(PDF)
proof=dict(status='targeted_audit_completed',input_pdf=str(PDF),input_sha256=sha(PDF),n_pages=len(reader.pages),
    source_pdf_unchanged=True,local_pdf_latex_source='not provided or located in referenced research workspace/Downloads MuSc files',
    scope='26 selected cells of Tables2/3/5/6/7, manually transcribed from inspected PDF pages8/9; not full paper numerical validation',
    latest_local_config=dict(backbone='ViT-L-14-336',input=518,seed=42,mvtec_rscin=[1,2,3],visa_rscin=[1,8,9]),
    counts={s:sum(r['status']==s for r in rows) for s in sorted({r['status'] for r in rows})},
    evidence_hashes={str(p):sha(p) for p in [RUN/'mean_metrics.csv',RUN/'benchmark_summary.csv',RUN/'verification.json']},
    audit_csv_sha256=sha(OUT/'selected_pdf_cells.csv'),source_sha256=sha(Path(__file__)))
(OUT/'verification.json').write_text(json.dumps(proof,indent=2))
print(json.dumps(proof['counts']))
