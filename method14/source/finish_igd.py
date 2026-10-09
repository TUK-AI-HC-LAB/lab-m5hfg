"""Verify preserved IGD scores and update the MuSc comparison table."""
import csv
import json
import subprocess
import sys
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from common import ROOT,OUT,RAW,CATEGORIES,sha,TAG

def read(path):
    with path.open() as f:return list(csv.DictReader(f))
env=json.loads((OUT/'environment.json').read_text())
assert env['status']=='completed'
rows=read(OUT/'category_metrics.csv');mean=read(OUT/'mean_metrics.csv')[0]
assert len(rows)==15 and sum(int(r['n_images']) for r in rows)==1725
audits=[]
for row in rows:
    category=row['category']
    for scale in [32,256]:
        marker=json.loads((OUT/f'p{scale}'/category/'training_complete.json').read_text())
        assert marker['status']=='completed' and sha(marker['checkpoint'])==marker['sha256']
    manifest=json.loads((OUT/category/'raw_manifest.json').read_text())
    assert sha(manifest['path'])==manifest['sha256']
    with np.load(manifest['path']) as z:
        labels,scores,maps,masks=[z[k] for k in ['image_labels','image_scores','anomaly_maps','masks']]
    assert maps.shape==masks.shape==(int(row['n_images']),256,256)
    assert np.isfinite(maps).all() and np.isfinite(scores).all()
    images=read(OUT/category/'image_scores.csv');assert len(images)==len(labels)
    for r,label,score in zip(images,labels,scores):
        assert sha(r['image_path'])==r['sha256'] and int(r['label'])==int(label) and float(r['score'])==float(score)
    metrics=dict(image_auroc=roc_auc_score(labels,scores),image_ap=average_precision_score(labels,scores),
        pixel_auroc=roc_auc_score(masks.ravel(),maps.ravel()),pixel_ap=average_precision_score(masks.ravel(),maps.ravel()))
    for k,v in metrics.items():assert abs(float(row[k])-float(v))<1e-12
    anomalous=[roc_auc_score(m.ravel(),p.ravel()) for y,m,p in zip(labels,masks,maps) if y and len(np.unique(m))==2]
    assert abs(float(row['official_style_anomaly_image_pixel_auroc'])-float(np.mean(anomalous)))<1e-12
    audits.append(dict(category=category,n_images=len(images),raw_metrics_verified=True,checkpoint_scales=[32,256]))
keys=['image_auroc','image_f1_max','image_ap','pixel_auroc','pixel_f1_max','pixel_ap','official_style_anomaly_image_pixel_auroc']
for k in keys:
    assert all(np.isfinite(float(r[k])) and -1e-12<=float(r[k])<=1+1e-12 for r in rows)
    assert abs(float(mean[k])-np.mean([float(r[k]) for r in rows]))<1e-12
(OUT/'verification.json').write_text(json.dumps(dict(status='passed',n_categories=15,n_images=1725,n_checkpoints=30,
    raw_metrics_recomputed=True,image_hashes_verified=True,macro_means_verified=True,categories=audits),indent=2))
md=ROOT.parent/'markdown/IGD_mvtec_execution.md'
text=md.read_text(encoding='utf-8').split('\n## 자체 측정 결과')[0]
lines=['','## 자체 측정 결과','','전체 학습·평가·검증 완료. 단위 %. Mean은 15개 category의 비가중 평균이며 단일 seed이므로 표준편차는 보고하지 않는다.','',
    '| Category | Images | '+' | '.join(keys)+' |','|---|---|'+'---|'*len(keys)]
for r in rows+[mean]:lines.append('| '+r['category']+' | '+r['n_images']+' | '+' | '.join(f'{100*float(r[k]):.4f}' for k in keys)+' |')
lines+=['','전체 픽셀을 합친 Pixel AUROC와 이상 이미지별 Pixel AUROC 평균은 서로 다른 집계다. MuSc Table 2에는 전체 픽셀 집계 값을 사용하고, 별도 열로 공식 평가 코드의 집계 방식도 보존한다.','',
    '[카테고리 CSV](../source/results/mvtec_full_seed42_20261007/category_metrics.csv) · '+
    '[평가 provenance](../source/results/mvtec_full_seed42_20261007/evaluation_provenance.json) · '+
    '[검증](../source/results/mvtec_full_seed42_20261007/verification.json)','',
    '원시 점수·맵에서 AUROC/AP와 공식 방식의 픽셀 집계를 재계산해 1e-12 이내 일치를 확인했다. 이 결과는 현재 PC에서 공식 학습 코드와 복구한 다중 스케일 평가 경로를 실행한 결과이며, 원문 Table 2의 저자 가중치·집계와의 동일성은 주장하지 않는다.','']
md.write_text((text+'\n'.join(lines)).replace('mvtec_full_seed42_20261007',TAG),encoding='utf-8')
subprocess.run([sys.executable,str(ROOT.parents[1]/'method10/source/build_musc_reproduction_tables.py')],check=True)
subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',
    r'C:\Users\test\Desktop\Codex\tools\update_obsidian_links.ps1'],check=True)
print('IGD ALL COMPLETE AND VERIFIED',flush=True)
