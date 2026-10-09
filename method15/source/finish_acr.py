"""Independently verify raw ACR predictions and update the measured-only table."""
import csv
import json
import subprocess
import sys
import numpy as np
from sklearn.metrics import average_precision_score,precision_recall_curve,roc_auc_score
from prepare_acr import ROOT,OUT,KEYS,sha
def read(path):
    with path.open() as f:return list(csv.DictReader(f))
def f1(y,s):
    p,r,_=precision_recall_curve(y,s)
    return float(np.max(np.divide(2*p*r,p+r,out=np.zeros_like(p),where=p+r>0)))
env=json.loads((OUT/'environment.json').read_text())
assert env['status']=='trained_and_evaluated'
rows=read(OUT/'category_metrics.csv');mean=read(OUT/'mean_metrics.csv')[0]
assert len(rows)==15 and sum(int(r['n_images']) for r in rows)==1725
audits=[]
for row in rows:
    folder=OUT/row['category']
    manifest=json.loads((folder/'raw_manifest.json').read_text())
    assert sha(manifest['path'])==manifest['sha256']
    checkpoint=json.loads((folder/'checkpoint.json').read_text())
    assert sha(checkpoint['path'])==checkpoint['sha256'] and checkpoint['iterations']==50
    history=read(folder/'training_history.csv')
    assert [int(r['iteration']) for r in history]==list(range(1,51))
    assert all(np.isfinite(float(r['loss'])) and r['finite_gradients']=='True' for r in history)
    with np.load(manifest['path']) as z:
        labels,scores,maps,masks=[z[k] for k in ['image_labels','image_scores','anomaly_maps','masks']]
    images=read(folder/'image_scores.csv')
    assert len(images)==len(labels)==int(row['n_images'])
    assert maps.shape==masks.shape==(len(labels),224,224) and np.isfinite(maps).all()
    assert np.array_equal(scores,maps.reshape(len(labels),-1).max(1))
    for r,y,s in zip(images,labels,scores):
        assert sha(r['image_path'])==r['sha256'] and int(r['label'])==int(y) and float(r['score'])==float(s)
    metrics=dict(image_auroc=roc_auc_score(labels,scores),image_ap=average_precision_score(labels,scores),image_f1_max=f1(labels,scores),
        pixel_auroc=roc_auc_score(masks.ravel(),maps.ravel()),pixel_ap=average_precision_score(masks.ravel(),maps.ravel()),pixel_f1_max=f1(masks.ravel(),maps.ravel()))
    for k,v in metrics.items():assert abs(float(row[k])-v)<1e-12
    native=json.loads((folder/'native_auroc.json').read_text())
    for k,v in native.items():assert abs(float(row[k])-v)<1e-12
    assert all(np.isfinite(float(row[k])) and -1e-12<=float(row[k])<=1+1e-12 for k in KEYS)
    audits.append(dict(category=row['category'],n_images=len(labels),raw_metrics_recomputed=True,native_auroc_matched=True))
for k in KEYS:assert abs(float(mean[k])-np.mean([float(r[k]) for r in rows]))<1e-12
(OUT/'verification.json').write_text(json.dumps(dict(status='passed',n_categories=15,n_images=1725,
    metrics_recomputed=['image_auroc','image_ap','image_f1_max','pixel_auroc','pixel_ap','pixel_f1_max'],
    aupro_verification='source hash and finite range; 200-threshold AUPRO is not independently recomputed',categories=audits),indent=2))
env['status']='completed';(OUT/'environment.json').write_text(json.dumps(env,indent=2))
md=ROOT.parent/'markdown/ACR_mvtec_execution.md'
text=md.read_text().split('\n## 자체 측정 결과')[0]
lines=['','## 자체 측정 결과','','15개 category, 전체 test 1,725장 완료. 수치는 %, Mean은 카테고리 비가중 평균이다. 단일 seed 실행으로 표준편차를 보고하지 않는다.','',
    '| Category | Images | '+' | '.join(KEYS)+' |','|---|---|'+'---|'*len(KEYS)]
for row in rows+[mean]:lines.append('| '+row['category']+' | '+row['n_images']+' | '+' | '.join(f'{100*float(row[k]):.4f}' for k in KEYS)+' |')
lines+=['','[카테고리 CSV](../source/results/mvtec_seed42_20261007/category_metrics.csv) · [검증](../source/results/mvtec_seed42_20261007/verification.json)','',
    '원시 맵·점수에서 AUROC/AP/F1-max 6개 지표를 독립적으로 재계산하고 1e-12 이내 일치를 확인했다. Image/Pixel AUROC는 공식 평가 함수 반환값과도 일치한다. AUPRO는 앞선 APRIL-GAN/DRAEM 비교와 같은 200-threshold 근사 구현이며 별도 재계산은 하지 않았다. 논문 성능 수치를 복사하지 않았고, 저자의 수치와 동등하다는 주장은 하지 않는다.','']
md.write_text(text+'\n'.join(lines))
subprocess.run([sys.executable,str(ROOT.parents[1]/'method10/source/build_musc_reproduction_tables.py')],check=True)
subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',r'C:\Users\test\Desktop\Codex\tools\update_obsidian_links.ps1'],check=True)
print('ACR ALL COMPLETE AND VERIFIED',flush=True)
