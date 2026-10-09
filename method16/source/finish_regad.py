"""Verify retained RegAD rounds, aggregate support variation and update tables."""
import csv
import json
import subprocess
import sys
import numpy as np
from sklearn.metrics import average_precision_score,precision_recall_curve,roc_auc_score
from prepare_regad import ROOT,OUT,KEYS,sha
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def write(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def classification(y,s):
    p,r,_=precision_recall_curve(y,s)
    return dict(image_auroc=roc_auc_score(y,s),image_f1_max=np.divide(2*p*r,p+r,out=np.zeros_like(p),where=p+r>0).max(),image_ap=average_precision_score(y,s))
env=json.loads((OUT/'environment.json').read_text());assert env['status']=='evaluated'
rows=read(OUT/'round_metrics.csv');rscin_rows=read(OUT/'rscin_round_metrics.csv')
assert len(rows)==450 and len(rscin_rows)==900
audits=[]
for row in rows:
    folder=OUT/f"{row['shot']}shot"/row['category'];round_id=int(row['round'])
    inputs=json.loads((folder/'inputs.json').read_text())
    assert sha(inputs['checkpoint'])==inputs['checkpoint_sha256'] and sha(inputs['support'])==inputs['support_sha256']
    manifest=json.loads((folder/f'round{round_id}_raw.json').read_text())
    assert sha(manifest['path'])==manifest['sha256']
    with np.load(manifest['path']) as z:
        y,s,maps,masks=[z[k] for k in ['image_labels','image_scores','anomaly_maps','masks']]
    images=read(folder/f'round{round_id}_images.csv')
    assert len(images)==len(y)==int(row['n_images']) and maps.shape==masks.shape==(len(y),224,224)
    assert np.isfinite(maps).all() and np.array_equal(s,maps.reshape(len(y),-1).max(1))
    for image,label,score in zip(images,y,s):
        assert sha(image['image_path'])==image['sha256'] and int(image['label'])==int(label) and float(image['score'])==float(score)
    for k,v in classification(y,s).items():assert abs(v-float(row[k]))<1e-12
    if round_id==0:
        for k,v in classification(masks.ravel(),maps.ravel()).items():assert abs(v-float(row[k.replace('image_','pixel_')]))<1e-12
    assert all(np.isfinite(float(row[k])) and -1e-12<=float(row[k])<=1+1e-12 for k in KEYS)
    audits.append(dict(shot=int(row['shot']),category=row['category'],round=round_id,n_images=len(y),pixel_metrics_recomputed=round_id==0))
categories=[];means=[];rscin_means=[]
for shot in [4,2,8]:
    selected=[r for r in rows if int(r['shot'])==shot]
    names=sorted(set(r['category'] for r in selected));assert len(names)==15
    for name in names:
        subset=[r for r in selected if r['category']==name];assert len(subset)==10
        categories.append(dict(dataset='mvtec',shot=shot,category=name,n_images=int(subset[0]['n_images']),n_rounds=10,
            **{k:float(np.mean([float(r[k]) for r in subset])) for k in KEYS},
            **{k+'_std':float(np.std([float(r[k]) for r in subset])) for k in KEYS}))
    macro_rounds=[{k:float(np.mean([float(r[k]) for r in selected if int(r['round'])==i])) for k in KEYS} for i in range(10)]
    assert sum(int(r['n_images']) for r in selected if int(r['round'])==0)==1725
    means.append(dict(dataset='mvtec',shot=shot,n_images=1725,n_categories=15,n_rounds=10,
        **{k:float(np.mean([r[k] for r in macro_rounds])) for k in KEYS},
        **{k+'_std':float(np.std([r[k] for r in macro_rounds])) for k in KEYS}))
    for flag in ['w/o','w']:
        subset=[r for r in rscin_rows if int(r['shot'])==shot and r['rscin']==flag]
        assert len(subset)==150
        rounds=[{k:np.mean([float(r[k]) for r in subset if int(r['round'])==i]) for k in KEYS[:3]} for i in range(10)]
        rscin_means.append(dict(dataset='mvtec',shot=shot,rscin=flag,n_rounds=10,**{k:float(np.mean([r[k] for r in rounds])) for k in KEYS[:3]},**{k+'_std':float(np.std([r[k] for r in rounds])) for k in KEYS[:3]}))
write(OUT/'category_metrics.csv',categories);write(OUT/'mean_metrics.csv',means);write(OUT/'rscin_mean_metrics.csv',rscin_means)
(OUT/'verification.json').write_text(json.dumps(dict(status='passed',n_conditions=3,n_categories_per_condition=15,n_rounds=450,
    image_metrics_recomputed_all_rounds=True,pixel_metrics_recomputed_first_round_per_category=True,
    aupro_verification='source hash and finite range; not independently recomputed',audits=audits),indent=2))
env['status']='completed_public_2_4_8shot';(OUT/'environment.json').write_text(json.dumps(env,indent=2))
md=ROOT.parent/'markdown/RegAD_mvtec_execution.md'
text=md.read_text().split('\n## 자체 측정 결과')[0]
lines=['','## 자체 측정 결과','','공식 public checkpoint를 이 PC에서 평가했다. 학습은 다시 수행하지 않았다. 수치는 %, 평균은 category 비가중 평균이다. ±는10개 고정 support round별 macro metric의 표준편차(ddof0)이며 모델 재학습 seed의 편차가 아니다.','',
    '| Shot | '+' | '.join(KEYS)+' |','|---|'+'---|'*len(KEYS)]
for row in means:lines.append('| '+str(row['shot'])+' | '+' | '.join(f"{100*row[k]:.4f} ± {100*row[k+'_std']:.4f}" for k in KEYS)+' |')
lines+=['','[카테고리 CSV](../source/results/mvtec_public_20261007/category_metrics.csv) · [검증](../source/results/mvtec_public_20261007/verification.json)','',
    '450회 원시 맵/이미지 SHA·라벨·점수와 공개 checkpoint/support SHA를 확인했다. Image3개 지표는 전체 round, Pixel3개 지표는 각 category·shot의 round0에서 독립 재계산했다. AUPRO는200 threshold 근사 구현이고 별도 재계산하지 않았다. 제한된 실제 특징에서 batch Mahalanobis 대조를 통과했으나 전체 출력의 bitwise 일치를 주장하지 않는다. 32-shot과 BTAD는 이 완료 범위에 포함되지 않는다.','']
md.write_text(text+'\n'.join(lines))
subprocess.run([sys.executable,str(ROOT.parents[1]/'method10/source/build_musc_reproduction_tables.py')],check=True)
subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',r'C:\Users\test\Desktop\Codex\tools\update_obsidian_links.ps1'],check=True)
print('REGAD PUBLIC 2/4/8-SHOT COMPLETE AND VERIFIED',flush=True)
