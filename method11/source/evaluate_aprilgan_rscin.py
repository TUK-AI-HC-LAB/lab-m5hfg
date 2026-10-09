"""Table 11: apply official MuSc Mobile_RsCIN to local APRIL-GAN scores."""
import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parent
RAW = Path('/home/test/aprilgan_results')
MODULE = Path('/home/test/MuSc/models/RsCIN_features/RsCIN.py')
spec = importlib.util.spec_from_file_location('official_mobile_rscin', MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))

def metrics(labels, scores):
    p, r, _ = precision_recall_curve(labels, scores)
    with np.errstate(divide='ignore', invalid='ignore'):
        f = 2*p*r/(p+r)
    return dict(image_auroc=float(roc_auc_score(labels, scores)),
                image_f1_max=float(np.max(f[np.isfinite(f)])),
                image_ap=float(average_precision_score(labels, scores)))

def evaluate(tag, feature_tag=None):
    if feature_tag is None:
        feature_tag = tag
    source = ROOT/'results'/tag
    feature_source = ROOT/'results'/feature_tag
    if not (source/'environment.json').exists() or not (feature_source/'environment.json').exists():
        return False
    env = json.loads((source/'environment.json').read_text())
    feature_env = json.loads((feature_source/'environment.json').read_text())
    if env['status']!='completed' or feature_env['status']!='completed':
        return False
    ds = 'visa' if tag.startswith('visa') else 'mvtec_ad'
    if tag.startswith('btad'):
        return False  # MuSc Table 11 does not contain BTAD.
    shared_ds='visa' if ds=='visa' else 'mvtec'
    shared=ROOT/'results'/f'rscin_shared_features_{shared_ds}_20261006'
    shared_state=shared/'environment.json'
    if not shared_state.exists() or json.loads(shared_state.read_text())['status']!='completed':
        subprocess.run(['/home/test/miniforge3/envs/patchcore-gpu/bin/python','-u',str(ROOT/'extract_shared_musc_features.py'),
                        '--dataset',shared_ds],check=True)
    destination = source/'rscin'
    destination.mkdir(exist_ok=True)
    rows=[]
    for original in read(source/'category_metrics.csv'):
        category=original['category']
        scores=read(source/category/'image_scores.csv')
        feature_scores=read(feature_source/category/'image_scores.csv')
        assert [(r['image_path'],r['label'],r['sha256']) for r in scores] == [
            (r['image_path'],r['label'],r['sha256']) for r in feature_scores]
        labels=np.array([int(r['label']) for r in scores])
        before=np.array([float(r['score']) for r in scores])
        feature_rows=read(shared/category/'images.csv')
        lookup={r['image_path']:i for i,r in enumerate(feature_rows)}
        feature_array=np.load(Path('/home/test/aprilgan_results/rscin_shared_musc_features')/shared_ds/category/'features.npy')
        for r in scores:
            entry=feature_rows[lookup[r['image_path']]]
            assert r['label']==entry['label'] and r['sha256']==entry['sha256']
        features=feature_array[[lookup[r['image_path']] for r in scores]]
        after=module.Mobile_RsCIN(before.copy(),dataset_name=ds,cls_tokens=features)
        assert np.isfinite(after).all()
        before_metrics=metrics(labels,before)
        for key,value in before_metrics.items():
            assert abs(value-float(original[key]))<1e-12,(tag,category,key)
        for flag,result in [('w/o',before_metrics),('w',metrics(labels,after))]:
            rows.append(dict(category=category,rscin=flag,n_images=len(labels),**result))
        with (destination/(category+'_scores.csv')).open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['image_path','label','score_before','score_after','sha256'])
            writer.writerows((r['image_path'],r['label'],x,y,r['sha256']) for r,x,y in zip(scores,before,after))
    names=list(rows[0])
    with (destination/'category_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader();writer.writerows(rows)
    macro=[]
    for flag in ['w/o','w']:
        selected=[r for r in rows if r['rscin']==flag]
        macro.append(dict(category='macro_mean',rscin=flag,n_images=sum(r['n_images'] for r in selected),
                          **{k:float(np.mean([r[k] for r in selected])) for k in ['image_auroc','image_f1_max','image_ap']}))
    with (destination/'mean_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader();writer.writerows(macro)
    (destination/'provenance.json').write_text(json.dumps(dict(source_tag=tag,alignment_tag=feature_tag,
        feature_tag=f'rscin_shared_features_{shared_ds}_20261006',
        official_source=str(MODULE),official_source_sha256=hashlib.sha256(MODULE.read_bytes()).hexdigest(),
        windows=[1,8,9] if ds=='visa' else [1,2,3],
        normalized_feature_dtype='float16; official MuSc in-place CUDA AMP normalization',
        shared_feature_source=str(shared),
        paper_features_used=False,status='completed'),indent=2))
    print('RsCIN completed:',tag,macro,flush=True)
    return True

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--feature-tag')
    args=p.parse_args()
    if not evaluate(args.tag,args.feature_tag):
        raise SystemExit('Run or local feature evidence not completed yet.')
