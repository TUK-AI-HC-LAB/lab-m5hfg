"""Apply official MuSc RsCIN to captured DRAEM image scores, verify alignment."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'results/mvtec_public_20261007'
SHARED=ROOT.parents[1]/'method11/source/results/rscin_shared_features_mvtec_20261006'
MODULE=Path('/home/test/MuSc/models/RsCIN_features/RsCIN.py')
spec=importlib.util.spec_from_file_location('official_rscin',MODULE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def read(path):
    with path.open() as f:return list(csv.DictReader(f))
def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def metrics(labels,scores):
    p,r,_=precision_recall_curve(labels,scores)
    f=np.divide(2*p*r,p+r,out=np.zeros_like(p),where=(p+r)!=0)
    return dict(image_auroc=float(roc_auc_score(labels,scores)),image_f1_max=float(f.max()),
                image_ap=float(average_precision_score(labels,scores)))

assert json.loads((SOURCE/'environment.json').read_text())['status']=='completed'
assert json.loads((SHARED/'environment.json').read_text())['status']=='completed'
DEST=SOURCE/'rscin';DEST.mkdir(exist_ok=True)
rows=[];alignment=[]
for native in read(SOURCE/'category_metrics.csv'):
    category=native['category']
    images=read(SOURCE/category/'image_scores.csv')
    reference=read(SHARED/category/'images.csv')
    lookup={r['image_path']:i for i,r in enumerate(reference)}
    assert len(images)==len(reference)==len(lookup)
    indices=[]
    for row in images:
        i=lookup[row['image_path']];ref=reference[i]
        assert row['label']==ref['label'] and row['sha256']==ref['sha256']
        assert hashlib.sha256(Path(row['image_path']).read_bytes()).hexdigest()==row['sha256']
        indices.append(i)
    path=Path('/home/test/aprilgan_results/rscin_shared_musc_features/mvtec')/category/'features.npy'
    features=np.load(path)[indices]
    before=np.array([float(r['score']) for r in images]);labels=np.array([int(r['label']) for r in images])
    after=module.Mobile_RsCIN(before.copy(),dataset_name='mvtec_ad',cls_tokens=features)
    assert np.isfinite(after).all()
    results={'w/o':metrics(labels,before),'w':metrics(labels,after)}
    for k,v in results['w/o'].items():assert abs(v-float(native[k]))<1e-12
    for flag,result in results.items():rows.append(dict(category=category,rscin=flag,n_images=len(images),**result))
    write(DEST/(category+'_scores.csv'),[dict(image_path=r['image_path'],label=r['label'],
        score_before=float(x),score_after=float(y),sha256=r['sha256']) for r,x,y in zip(images,before,after)])
    alignment.append(dict(category=category,n_images=len(images),labels_and_sha256_match=True,
        feature_reordered=indices!=list(range(len(indices))),feature_path=str(path),
        feature_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
write(DEST/'category_metrics.csv',rows)
macro=[]
for flag in ['w/o','w']:
    selected=[r for r in rows if r['rscin']==flag]
    macro.append(dict(category='macro_mean',rscin=flag,n_images=sum(r['n_images'] for r in selected),
        **{k:float(np.mean([r[k] for r in selected])) for k in ['image_auroc','image_f1_max','image_ap']}))
write(DEST/'mean_metrics.csv',macro);write(DEST/'dataset_alignment.csv',alignment)
(DEST/'provenance.json').write_text(json.dumps(dict(status='completed',official_source=str(MODULE),
    official_source_sha256=hashlib.sha256(MODULE.read_bytes()).hexdigest(),shared_feature_source=str(SHARED),
    shared_feature_environment_sha256=hashlib.sha256((SHARED/'environment.json').read_bytes()).hexdigest(),
    windows=[1,2,3],feature_dtype='float16; normalized in official MuSc CUDA AMP',
    feature_encoder='ViT-L-14-336 OpenAI; input 518; official MuSc loader',
    paper_features_used=False,training_executed=False,n_categories=15,n_images=1725),indent=2))
print('RsCIN COMPLETE',macro,flush=True)
