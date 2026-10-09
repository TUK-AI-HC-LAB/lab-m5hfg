"""Verify finished local NSA results, render report, and update MuSc Table 2."""
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/logistic_seed923874273_20261007'
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
env=json.loads((OUT/'environment.json').read_text())
assert env['status']=='completed'
rows=read(OUT/'category_metrics.csv')
assert len(rows)==15 and sum(int(r['n_images']) for r in rows)==1725
checks=[]
for r in rows:
    category=r['category'];folder=OUT/category
    train=json.loads((folder/'training_complete.json').read_text())
    assert sha(train['checkpoint'])==train['sha256']
    manifest=json.loads((folder/'raw_manifest.json').read_text())
    assert sha(manifest['path'])==manifest['sha256']
    with np.load(manifest['path']) as z:
        labels,scores,masks,maps=[z[k] for k in ['image_labels','image_scores','masks','anomaly_maps']]
    images=read(folder/'image_scores.csv')
    assert len(images)==len(labels)==int(r['n_images'])
    for y,s,img in zip(labels,scores,images):
        assert int(img['label'])==int(y) and float(img['score'])==float(s)
        assert sha(img['image_path'])==img['sha256']
    native=dict(image_auroc=roc_auc_score(labels,scores),image_ap=average_precision_score(labels,scores),
                pixel_auroc=roc_auc_score(masks.ravel(),maps.ravel()),pixel_ap=average_precision_score(masks.ravel(),maps.ravel()))
    assert all(abs(float(r[k])-float(v))<1e-12 for k,v in native.items())
    checks.append(dict(category=category,n_images=len(images),native_metrics_verified=True))
mean=read(OUT/'mean_metrics.csv')[0]
keys=['image_auroc','image_f1_max','image_ap','pixel_auroc','pixel_f1_max','pixel_ap','aupro']
for k in keys:assert abs(float(mean[k])-np.mean([float(r[k]) for r in rows]))<1e-12
(OUT/'verification.json').write_text(json.dumps(dict(status='passed',n_categories=15,n_images=1725,
    checkpoint_hashes_verified=True,native_metrics_recomputed=True,categories=checks),indent=2))
md=ROOT.parent/'markdown/NSA_mvtec_execution.md'
text=md.read_text(encoding='utf-8').split('\n## 자체 측정 결과')[0]
lines=['','## 자체 측정 결과','','15개 category, 1,725장 전체 학습·평가 완료. 단위 %. Mean은 category 비가중 평균. 단일 seed로 반복 표준편차를 보고하지 않는다.','',
    '| Category | Images | '+' | '.join(keys)+' |','|---|---|'+'---|'*len(keys)]
for r in rows+[mean]:lines.append('| '+r['category']+' | '+r['n_images']+' | '+' | '.join(f'{100*float(r[k]):.4f}' for k in keys)+' |')
lines+=['','[환경](../source/results/logistic_seed923874273_20261007/environment.json) · '+
    '[검증](../source/results/logistic_seed923874273_20261007/verification.json) · '+
    '[카테고리 CSV](../source/results/logistic_seed923874273_20261007/category_metrics.csv)','',
    '공식 평가 함수의 AUROC/AP 4개 지표를 보존된 원시 NPZ에서 다시 계산해 1e-12 이내 일치를 검증했다. 공식 NSA AUPRO는 FPR 0.3에서 보간·정규화하는 구현이며, 다른 방법에서 사용한 200 threshold 근사 AUPRO와 구현 차이가 있다. 학습 seed, worker 수 및 현대 의존성은 이 PC의 실행 조건으로 기록하며, 저자 수치 동등성을 주장하지 않는다.','']
md.write_text(text+'\n'.join(lines),encoding='utf-8')
subprocess.run([sys.executable,str(ROOT.parents[1]/'method10/source/build_musc_reproduction_tables.py')],check=True)
subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',
    r'C:\Users\test\Desktop\Codex\tools\update_obsidian_links.ps1'],check=True)
print('NSA ALL COMPLETE AND VERIFIED',flush=True)
