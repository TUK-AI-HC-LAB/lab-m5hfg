"""Verify raw native predictions, aggregate and update MuSc tables."""
import json
import subprocess
import sys
import numpy as np
from graphcore_common import *

env=json.loads((OUT/'environment.json').read_text());assert env['status']=='evaluated'
subprocess.run([sys.executable,str(ROOT/'check_cache.py')],check=True)
cache_proof=json.loads((OUT/'native_cache_comparison.json').read_text());assert cache_proof['status']=='passed'
rows=read(OUT/'round_metrics.csv');assert len(rows)==300
assert sha(WEIGHT)==env['weight_sha256']
audits=[]
for row in rows:
    shot=int(row['shot']);r=int(row['round']);cat=row['category'];folder=OUT/cat
    manifest=json.loads((folder/f'{shot}shot_round{r}_raw.json').read_text())
    assert sha(manifest['path'])==manifest['sha256'] and sha(manifest['bank_path'])==manifest['bank_sha256']
    bank=np.load(manifest['bank_path']);assert bank.shape==(int(shot*784*.01),336)
    with np.load(manifest['path']) as z:y,s,maps,masks=[z[k] for k in ['image_labels','image_scores','anomaly_maps','masks']]
    images=read(folder/'images.csv');scores=read(folder/f'{shot}shot_round{r}_scores.csv')
    assert len(images)==len(scores)==len(y)==int(row['n_images']) and maps.shape==masks.shape==(len(y),224,224)
    for image,score,label,value in zip(images,scores,y,s):
        assert image['image_path']==score['image_path'] and int(image['label'])==int(label)==int(score['label']) and float(score['score'])==float(value)
        if r==0:
            assert sha(image['image_path'])==image['sha256']
            if image['mask_path']:assert sha(image['mask_path'])==image['mask_sha256']
    for k,v in classification(y,s).items():assert abs(v-float(row[k]))<1e-12
    if r==0:
        for k,v in classification(masks.ravel(),maps.ravel()).items():assert abs(v-float(row[k.replace('image_','pixel_')]))<1e-12
    assert np.isfinite(maps).all() and all(np.isfinite(float(row[k])) and -1e-12<=float(row[k])<=1+1e-12 for k in KEYS)
    audits.append(dict(shot=shot,category=cat,round=r,n_images=len(y),pixel_recomputed=r==0))
categories=[];means=[]
for shot in [4,8]:
    selected=[r for r in rows if int(r['shot'])==shot]
    names=sorted(set(r['category'] for r in selected));assert len(names)==15
    for cat in names:
        sub=[r for r in selected if r['category']==cat];assert len(sub)==10
        categories.append(dict(dataset='mvtec',shot=shot,category=cat,n_images=int(sub[0]['n_images']),n_rounds=10,
            **{k:float(np.mean([float(r[k]) for r in sub])) for k in KEYS},**{k+'_std':float(np.std([float(r[k]) for r in sub])) for k in KEYS}))
    rounds=[{k:float(np.mean([float(r[k]) for r in selected if int(r['round'])==i])) for k in KEYS} for i in range(10)]
    assert sum(int(r['n_images']) for r in selected if int(r['round'])==0)==1725
    means.append(dict(dataset='mvtec',shot=shot,n_categories=15,n_images=1725,n_rounds=10,
        **{k:float(np.mean([r[k] for r in rounds])) for k in KEYS},**{k+'_std':float(np.std([r[k] for r in rounds])) for k in KEYS}))
write(OUT/'category_metrics.csv',categories);write(OUT/'mean_metrics.csv',means)
(OUT/'verification.json').write_text(json.dumps(dict(status='passed',n_conditions=2,n_categories=15,n_rounds=300,
    native_train_prediction_methods=True,cache_comparison=cache_proof,image_metrics_recomputed_all_rounds=True,pixel_metrics_recomputed_round0=True,
    aupro_verification='source SHA and range; not independently recomputed',audits=audits),indent=2))
env['status']='completed';(OUT/'environment.json').write_text(json.dumps(env,indent=2))
md=ROOT.parent/'markdown/GraphCore_mvtec_execution.md'
text=md.read_text(encoding='utf-8').split('\n## 자체 측정 결과')[0]
lines=['','## 자체 측정 결과','','단위는 %. ±는 10개 support round별 15-category 비가중 평균의 표준편차(ddof0)이다.','',
    '| Shot | '+' | '.join(KEYS)+' |','|---|'+'---|'*7]
for row in means:lines.append('| '+str(row['shot'])+' | '+' | '.join(f"{100*row[k]:.4f} ± {100*row[k+'_std']:.4f}" for k in KEYS)+' |')
lines+=['','[카테고리 CSV](../source/results/mvtec_pvig_fp32_20261008/category_metrics.csv) · [검증 JSON](../source/results/mvtec_pvig_fp32_20261008/verification.json)','']
md.write_text(text+'\n'.join(lines),encoding='utf-8')
subprocess.run([sys.executable,str(ROOT.parents[1]/'method10/source/build_musc_reproduction_tables.py')],check=True)
subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',r'C:\Users\test\Desktop\Codex\tools\update_obsidian_links.ps1'],check=True)
print('GRAPHCORE 4/8SHOT COMPLETE AND VERIFIED',flush=True)
