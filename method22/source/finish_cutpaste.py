"""Validate all 15 categories, both training runs, raw scores and AUROC."""
import json,subprocess,sys
import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from cutpaste_common import *
from evaluate_cutpaste import kernel,maps_from_scores
env=json.loads((OUT/'environment.json').read_text());assert env['steps_per_model']==65536 and env['no_pretraining']
assert sha(ROOT/'run_cutpaste.py')==env['wrapper_sha256'] and sha(ROOT/'cutpaste_common.py')==env['common_sha256']
assert sha(ROOT/'cutpaste_acceleration.py')==env['acceleration_helper_sha256']
for name,digest in env['source_hashes'].items():assert sha(REPO/name)==digest
rows=read(OUT/'category_metrics.csv');assert [r['category'] for r in rows]==CATS and sum(int(r['n_images']) for r in rows)==1725
audits=[]
for row in rows:
    cat=row['category'];out=OUT/cat;raw=RAW/cat;tr=read(out/'train_images.csv');te=read(out/'test_images.csv')
    assert len(te)==int(row['n_images']) and len(tr)==int(row['n_train'])
    assert set(r['image_path'] for r in tr).isdisjoint(r['image_path'] for r in te)
    for r in tr+te:
        assert sha(r['image_path'])==r['sha256']
        if r['mask_path']:assert sha(r['mask_path'])==r['mask_sha256']
    reference=read(ROOT.parents[1]/f'method10/source/results/mvtec_all_paper_20261002/{cat}/image_scores.csv')
    assert {r['image_path']:(int(r['label']),r['sha256']) for r in reference}=={r['image_path']:(int(r['label']),r['sha256']) for r in te}
    for art in json.loads((out/'raw_manifest.json').read_text()):assert sha(art['path'])==art['sha256']
    for kind in ['image','patch']:
        h=read(out/kind/'training.csv');assert [int(r['step']) for r in h]==list(range(256,STEPS+1,256))
        assert all(np.isfinite(float(r['loss'])) for r in h)
        cp=json.loads((out/kind/'checkpoint.json').read_text());assert sha(cp['path'])==cp['sha256']
        saved=torch.load(cp['path'],map_location='cpu',weights_only=False);assert saved['step']==STEPS and not saved['config']['pretrained'];del saved
        assert json.loads((out/kind/'first_gpu_step.json').read_text())['status']=='passed'
    assert json.loads((out/'density_verification.json').read_text())['status']=='passed'
    with np.load(raw/'predictions.npz') as z:labels,scores,patch,maps,ms=[z[k] for k in ['labels','image_scores','patch_scores','score_maps','masks']]
    assert np.array_equal(labels,np.array([int(r['label']) for r in te])) and np.array_equal(ms,masks(te))
    assert np.isfinite(scores).all() and np.isfinite(maps).all()
    train_feats=np.load(raw/'image/train_features.npy');test_feats=np.load(raw/'image/test_features.npy')
    density=LedoitWolf().fit(np.asarray(train_feats,dtype=np.float64));q=np.asarray(test_feats,dtype=np.float64)-density.location_
    expected=np.maximum(np.einsum('bi,ij,bj->b',q,density.precision_,q),0);assert np.allclose(expected,scores,rtol=1e-9,atol=1e-9)
    pfeat=np.load(raw/'patch/test_features.npy',mmap_mode='r');mu=np.load(raw/'patch/density_mean.npy',mmap_mode='r');inv=np.load(raw/'patch/density_precision.npy',mmap_mode='r')
    aligned=cat in ALIGNED
    for i in [0,len(te)-1]:
        for j in [0,1624,3248]:
            q=pfeat[i,j].astype(np.float64)-mu[j if aligned else 0];value=max(float(q@inv[j if aligned else 0]@q),0)
            assert np.allclose(value,patch[i,j],rtol=1e-7,atol=1e-6)
    # All maps reprocessed, plus independently sum first map's Gaussian RFs.
    assert np.array_equal(maps_from_scores(patch),maps)
    independent=np.zeros((256,256),dtype=np.float64);k=kernel().numpy()[0,0]
    for h in range(57):
        for w in range(57):independent[h*4:h*4+32,w*4:w*4+32]+=patch[0,h*57+w]*k
    assert np.allclose(independent,maps[0],rtol=1e-5,atol=1e-4)
    image_auc=roc_auc_score(labels,scores);pixel_auc=roc_auc_score(ms.ravel(),maps.ravel())
    assert abs(image_auc-float(row['image_auroc']))<1e-12 and abs(pixel_auc-float(row['pixel_auroc']))<1e-12
    audits.append(dict(category=cat,n_train=len(tr),n_test=len(te),image_auroc=float(image_auc),pixel_auroc=float(pixel_auc),
        training_steps_per_model=STEPS,independent_RF_first_map_max_error=float(np.max(np.abs(independent-maps[0])))))
mean=dict(dataset='mvtec',n_categories=15,n_images=1725,image_auroc=float(np.mean([float(r['image_auroc']) for r in rows])),pixel_auroc=float(np.mean([float(r['pixel_auroc']) for r in rows])))
write(OUT/'mean_metrics.csv',[mean])
(OUT/'verification.json').write_text(json.dumps(dict(status='passed',all_input_raw_checkpoint_hashes_checked=True,official_MuSc_image_labels_SHA_matched=True,
    all_AUROC_recomputed=True,all_global_GDE_scores_refit=True,all_RF_maps_reprocessed=True,
    limits='patch density formula checked on selected actual features; patch scores sampled, all dense model forwards not rerun. No paper numerical equivalence claim.',audits=audits),indent=2))
env['status']='completed';(OUT/'environment.json').write_text(json.dumps(env,indent=2))
md=ROOT.parent/'markdown/CutPaste_mvtec_execution.md';text=md.read_text(encoding='utf-8').split('\n## 자체 측정 결과')[0]
lines=['','## 자체 측정 결과','','15개 카테고리 비가중 평균, 단위 %. 단일 seed이며 논문의 5회 평균·SE와 구분한다.','',
    '| Category | Test images | Image AUROC | Pixel AUROC |','|---|---|---|---|']
for r in rows:lines.append(f"| {r['category']} | {r['n_images']} | {100*float(r['image_auroc']):.4f} | {100*float(r['pixel_auroc']):.4f} |")
lines += [f"| Mean | 1725 | {100*mean['image_auroc']:.4f} | {100*mean['pixel_auroc']:.4f} |",'',
    '[검증](../source/results/mvtec_scratch3way_seed42_20261008/verification.json) · [CSV](../source/results/mvtec_scratch3way_seed42_20261008/category_metrics.csv)','']
md.write_text(text+'\n'.join(lines),encoding='utf-8')
subprocess.run([sys.executable,str(ROOT.parents[1]/'method10/source/build_musc_reproduction_tables.py')],check=True)
subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',r'C:\Users\test\Desktop\Codex\tools\update_obsidian_links.ps1'],check=True)
print('CUTPASTE MVTec VERIFIED COMPLETE',flush=True)
