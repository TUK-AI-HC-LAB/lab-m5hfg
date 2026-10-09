"""Independent raw/input/metric checks and an independent pyramid example."""
import json,subprocess,sys
import numpy as np
import torch
from scipy.ndimage import correlate1d
from sklearn.metrics import roc_auc_score
from pyramidflow_common import *

env=json.loads((OUT/'environment.json').read_text());assert env['epochs']==15 and env['batch_size']==2
assert env['buffer_fix_verification']['status']=='passed'
assert sha(env['weights_path'])==env['weights_sha256'] and sha(DEPENDENCY/'autoFlow.py')==env['dependency_sha256']
for name,digest in env['source_hashes'].items():assert sha(REPO/name)==digest
assert sha(ROOT/'run_pyramidflow.py')==env['wrapper_sha256'] and sha(ROOT/'pyramidflow_common.py')==env['common_sha256']
reference=json.loads((DATA/'meta.json').read_text());rows=read(OUT/'category_metrics.csv');assert [r['category'] for r in rows]==['01','02','03']
audits=[]
for row in rows:
    cat=row['category'];folder=OUT/cat;manifest_raw=json.loads((folder/'raw_manifest.json').read_text())
    for artifact in manifest_raw.values():assert sha(artifact['path'])==artifact['sha256']
    train=read(folder/'train_images.csv');test=read(folder/'test_images.csv');assert len(test)==int(row['n_images']) and len(train)==int(row['n_train'])
    assert set(r['image_path'] for r in train).isdisjoint(r['image_path'] for r in test)
    for phase,items in [('train',train),('test',test)]:
        lookup={str(DATA/r['img_path']):r for r in reference[phase][cat]};assert set(lookup)==set(r['image_path'] for r in items)
        for r in items:
            assert sha(r['image_path'])==r['sha256']
            if phase=='test':
                ref=lookup[r['image_path']];assert int(r['label'])==int(ref['anomaly'])
                if r['mask_path']:assert r['mask_path']==str(DATA/ref['mask_path']) and sha(r['mask_path'])==r['mask_sha256']
    with np.load(manifest_raw['predictions.npz']['path']) as z:
        labels,masks,maps,scores=[z[k] for k in ['labels','masks','score_maps','image_scores']]
    assert np.array_equal(labels,np.array([int(r['label']) for r in test]))
    assert np.array_equal(masks,inputs(test)) and np.isfinite(maps).all()
    assert np.array_equal(scores,maps.max(axis=(1,2)))
    score_rows=read(folder/'image_scores.csv');assert len(score_rows)==len(test)
    assert all(float(r['score'])==float(s) and r['image_path']==t['image_path'] for r,s,t in zip(score_rows,scores,test))
    image_auc=roc_auc_score(labels,scores);pixel_auc=roc_auc_score(masks.ravel(),maps.ravel())
    assert abs(image_auc-float(row['image_auroc']))<1e-12 and abs(pixel_auc-float(row['pixel_auroc']))<1e-12
    history=read(folder/'training.csv');assert [int(r['epoch']) for r in history]==list(range(1,16))
    assert all(int(r['n_batches'])==len(train)//2 and np.isfinite(float(r['loss'])) for r in history)
    saved=torch.load(manifest_raw['last.pt']['path'],map_location='cpu',weights_only=True);assert saved['epoch']==15 and 'optimizer' in saved
    del saved
    proof=json.loads((folder/'first_gpu_step.json').read_text());assert proof['status']=='passed' and proof['finite_gradients']
    example=torch.load(manifest_raw['first_test_latent.pt']['path'],map_location='cpu',weights_only=True)
    template=torch.load(manifest_raw['template.pt']['path'],map_location='cpu',weights_only=True)
    assert all(torch.equal(a,b) for a,b in zip(example['template'],template))
    diff=[np.abs(a.numpy()-b.numpy()) for a,b in zip(example['latent'],template)]
    composed=diff[-1];kernel=np.array([1,4,6,4,1],dtype=np.float32)/16
    for level in reversed(diff[:-1]):
        # Independent nearest-neighbor upsampling and separable Gaussian
        # convolution with constant-zero border, native binomial 5x5 kernel.
        composed=np.repeat(np.repeat(composed,2,axis=-2),2,axis=-1)
        composed=correlate1d(correlate1d(composed,kernel,axis=-2,mode='constant',cval=0),kernel,axis=-1,mode='constant',cval=0)+level
    reproduced=composed.mean(axis=1)[0]
    assert np.allclose(reproduced,maps[0],rtol=1e-4,atol=1e-4)
    audits.append(dict(category=cat,n_train=len(train),n_test=len(test),epochs=15,image_auroc=float(image_auc),pixel_auroc=float(pixel_auc),
        first_map_independent_pyramid_max_error=float(np.max(np.abs(reproduced-maps[0]))),first_gpu_check=proof,
        raw_sha256=manifest_raw['predictions.npz']['sha256']))
mean=dict(dataset='btad',n_categories=3,n_images=sum(int(r['n_images']) for r in rows),
    image_auroc=float(np.mean([float(r['image_auroc']) for r in rows])),pixel_auroc=float(np.mean([float(r['pixel_auroc']) for r in rows])))
assert mean['n_images']==741
write(OUT/'mean_metrics.csv',[mean])
(OUT/'verification.json').write_text(json.dumps(dict(status='passed',all_raw_checkpoint_input_hashes_checked=True,all_masks_and_AUROC_recomputed=True,
    all_image_max_scores_recomputed=True,first_pixel_map_each_category_independent_compose_checked=True,
    limitation='does not independently rerun all test forwards or prove paper numerical equivalence',audits=audits),indent=2))
env['status']='completed';(OUT/'environment.json').write_text(json.dumps(env,indent=2))
md=ROOT.parent/'markdown/PyramidFlow_btad_execution.md';text=md.read_text(encoding='utf-8').split('\n## 자체 측정 결과')[0]
lines=['','## 자체 측정 결과','','제품 3개 비가중 평균, 단위 %. 마지막 epoch15의 모델이며 best-test 선택은 하지 않았다.','',
    '| Product | Test images | Image AUROC | Pixel AUROC |','|---|---|---|---|']
for r in rows:lines.append(f"| {r['category']} | {r['n_images']} | {100*float(r['image_auroc']):.4f} | {100*float(r['pixel_auroc']):.4f} |")
lines += [f"| Mean | {mean['n_images']} | {100*mean['image_auroc']:.4f} | {100*mean['pixel_auroc']:.4f} |",'',
    '[검증](../source/results/btad_res18_seed0_20261008/verification.json) · [CSV](../source/results/btad_res18_seed0_20261008/category_metrics.csv)','']
md.write_text(text+'\n'.join(lines),encoding='utf-8')
subprocess.run([sys.executable,str(ROOT.parents[1]/'method10/source/build_musc_reproduction_tables.py')],check=True)
subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',r'C:\Users\test\Desktop\Codex\tools\update_obsidian_links.ps1'],check=True)
print('PYRAMIDFLOW BTAD VERIFIED COMPLETE',flush=True)
