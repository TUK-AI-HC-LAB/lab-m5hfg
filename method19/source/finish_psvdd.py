"""Verify raw native multi-scale predictions and update MuSc Table14."""
import sys,json,subprocess
import numpy as np
from psvdd_common import *
env=json.loads((OUT/'environment.json').read_text());assert env['status']=='evaluated'
proof=json.loads((OUT/'native_runtime_verification.json').read_text());assert proof['status']=='passed'
rows=read(OUT/'category_metrics.csv');assert len(rows)==3 and {r['category'] for r in rows}=={'01','02','03'}
reference=json.loads((DATA/'meta.json').read_text());audits=[]
for row in rows:
    cat=row['category'];folder=OUT/cat
    manifest=json.loads((folder/'raw_manifest.json').read_text())
    assert sha(manifest['path'])==manifest['sha256'] and sha(manifest['checkpoint'])==manifest['checkpoint_sha256']
    with np.load(manifest['path']) as z:
        labels,masks,m64,m32,msum,mmult=[z[k] for k in ['image_labels','masks','maps_64','maps_32','maps_sum','maps_mult']]
    assert np.array_equal(msum,m64+m32) and np.array_equal(mmult,m64*m32)
    assert mmult.shape==masks.shape==(len(labels),256,256) and np.isfinite(mmult).all()
    scores=mmult.reshape(len(labels),-1).max(1);images=read(folder/'test_images.csv');train=read(folder/'train_images.csv');scores_csv=read(folder/'image_scores.csv')
    assert len(images)==len(labels)==int(row['n_images'])
    assert set(r['image_path'] for r in train).isdisjoint(r['image_path'] for r in images)
    # Reference metadata has sorted interleaved ok/ko; native evaluation puts
    # anomalies first. Match paths rather than assuming the same row order.
    lookup={str(DATA/r['img_path']):r for r in reference['test'][cat]}
    assert set(lookup)==set(r['image_path'] for r in images)
    for image,label,score,saved in zip(images,labels,scores,scores_csv):
        ref=lookup[image['image_path']]
        assert sha(image['image_path'])==image['sha256'] and int(image['label'])==int(label)==int(ref['anomaly'])==int(saved['label'])
        assert saved['image_path']==image['image_path'] and float(saved['score'])==float(score)
        if image['mask_path']:assert str(DATA/ref['mask_path'])==image['mask_path'] and sha(image['mask_path'])==image['mask_sha256']
    assert set(str(DATA/r['img_path']) for r in reference['train'][cat])==set(r['image_path'] for r in train)
    for image in train:assert sha(image['image_path'])==image['sha256'] and int(image['label'])==0
    for k,v in metrics(labels,scores,masks,mmult).items():assert abs(v-float(row[k]))<1e-12
    history=read(folder/'training.csv');assert len(history)==300 and int(history[0]['n_batches'])==0
    assert all(int(r['n_batches'])==int(np.ceil(len(train)*100/64)) for r in history[1:]) and int(row['checkpoint_epoch'])==300
    assert json.loads((folder/'first_gpu_step.json').read_text())['status']=='passed'
    audits.append(dict(category=cat,n_train=len(train),n_test=len(labels),epoch_slots=300,update_epochs=299,raw_sha256=manifest['sha256']))
mean=dict(dataset='btad',n_categories=3,n_images=sum(int(r['n_images']) for r in rows),fusion='mult',
    image_auroc=float(np.mean([float(r['image_auroc']) for r in rows])),pixel_auroc=float(np.mean([float(r['pixel_auroc']) for r in rows])))
assert mean['n_images']==741
write(OUT/'mean_metrics.csv',[mean])
(OUT/'verification.json').write_text(json.dumps(dict(status='passed',all_raw_and_input_hashes_checked=True,all_AUROC_recomputed=True,fusion_recomputed=True,native_runtime_check=proof,audits=audits),indent=2))
env['status']='completed';(OUT/'environment.json').write_text(json.dumps(env,indent=2))
md=ROOT.parent/'markdown/P_SVDD_btad_execution.md';text=md.read_text(encoding='utf-8').split('\n## 자체 측정 결과')[0]
lines=['','## 자체 측정 결과','','수치는 %, 제품3개 비가중 평균이다. seed42단일 학습이며 ±를 붙이지 않는다.','',
    '| Product | Test images | Image AUROC | Pixel AUROC |','|---|---|---|---|']
for r in rows:lines.append(f"| {r['category']} | {r['n_images']} | {100*float(r['image_auroc']):.4f} | {100*float(r['pixel_auroc']):.4f} |")
lines += [f"| Mean | {mean['n_images']} | {100*mean['image_auroc']:.4f} | {100*mean['pixel_auroc']:.4f} |",'',
    '[검증](../source/results/btad_seed42_20261008/verification.json) · [CSV](../source/results/btad_seed42_20261008/category_metrics.csv)','']
md.write_text(text+'\n'.join(lines),encoding='utf-8')
subprocess.run([sys.executable,str(ROOT.parents[1]/'method10/source/build_musc_reproduction_tables.py')],check=True)
subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',r'C:\Users\test\Desktop\Codex\tools\update_obsidian_links.ps1'],check=True)
print('P-SVDD BTAD VERIFIED COMPLETE',flush=True)
