"""Validate raw predictions and input identity before publishing results."""
import json,subprocess,sys
import numpy as np
from sklearn.metrics import roc_auc_score
from spade_common import *
from run_spade import map256

env=json.loads((OUT/'environment.json').read_text());assert env['K']==50 and env['training_epochs']==0
assert sha(env['weights_path'])==env['weights_sha256']
assert sha(ROOT/'run_spade.py')==env['wrapper_sha256'] and sha(ROOT/'spade_common.py')==env['common_sha256']
for path,digest in env['source_hashes'].items():assert sha(REPO/path)==digest
reference=json.loads((DATA/'meta.json').read_text());rows=read(OUT/'category_metrics.csv');assert [r['category'] for r in rows]==['01','02','03']
audits=[]
for row in rows:
    cat=row['category'];folder=OUT/cat;proof=json.loads((folder/'distance_verification.json').read_text())
    assert proof['status']=='passed' and proof['gallery_size']==50*56*56
    manifest_raw=json.loads((folder/'raw_manifest.json').read_text());assert sha(manifest_raw['path'])==manifest_raw['sha256']
    for f in manifest_raw['global_features'].values():assert sha(f['path'])==f['sha256']
    raw=np.load(manifest_raw['path']);test=read(folder/'test_images.csv');train=read(folder/'train_images.csv')
    assert len(test)==int(row['n_images']) and len(train)==int(row['n_train'])
    assert set(r['image_path'] for r in train).isdisjoint(r['image_path'] for r in test)
    for phase,inputs in [('train',train),('test',test)]:
        lookup={str(DATA/r['img_path']):r for r in reference[phase][cat]}
        assert set(lookup)==set(r['image_path'] for r in inputs)
        for r in inputs:
            assert sha(r['image_path'])==r['sha256']
            if phase=='test':
                ref=lookup[r['image_path']];assert int(r['label'])==int(ref['anomaly'])
                if r['mask_path']:assert r['mask_path']==str(DATA/ref['mask_path']) and sha(r['mask_path'])==r['mask_sha256']
    labels=np.array([int(r['label']) for r in test]);assert np.array_equal(raw['labels'],labels)
    expected_masks=masks(test);assert np.array_equal(expected_masks,raw['masks'])
    assert np.isfinite(raw['score_maps']).all() and np.isfinite(raw['image_scores']).all()
    tg=np.load(manifest_raw['global_features']['train']['path']).reshape(len(train),-1).astype(np.float64)
    qg=np.load(manifest_raw['global_features']['test']['path']).reshape(len(test),-1).astype(np.float64)
    reference_scores=[];maxerr=0.
    for i,q in enumerate(qg):
        delta=tg-q;d=np.einsum('ij,ij->i',delta,delta);selected=raw['neighbor_indices'][i]
        assert len(np.unique(selected))==50
        assert np.allclose(np.sort(d[selected]),np.sort(d)[:50],rtol=1e-5,atol=1e-4)
        reference_scores.append(np.sort(d)[:50].mean())
        reproduced=map256(raw['nn56'][i]);assert np.array_equal(reproduced,raw['score_maps'][i])
    assert np.allclose(raw['image_scores'],reference_scores,rtol=1e-5,atol=1e-4)
    scores_csv=read(folder/'image_scores.csv')
    assert all(float(r['score'])==float(s) and r['image_path']==t['image_path'] for r,s,t in zip(scores_csv,raw['image_scores'],test))
    image_auc=float(roc_auc_score(labels,raw['image_scores']));pixel_auc=float(roc_auc_score(expected_masks.ravel(),raw['score_maps'].ravel()))
    assert abs(image_auc-float(row['image_auroc']))<1e-12 and abs(pixel_auc-float(row['pixel_auroc']))<1e-12
    audits.append(dict(category=cat,n_train=len(train),n_test=len(test),raw_sha256=manifest_raw['sha256'],
        independent_float64_image_score_max_error=float(np.max(np.abs(np.array(reference_scores)-raw['image_scores']))),
        all_K50_global_neighbors_verified=True,all_pixel_postprocessing_recomputed=True,dense_distance_check=proof))
mean=dict(dataset='btad',n_categories=3,n_images=sum(int(r['n_images']) for r in rows),
    image_auroc=float(np.mean([float(r['image_auroc']) for r in rows])),pixel_auroc=float(np.mean([float(r['pixel_auroc']) for r in rows])))
assert mean['n_images']==741
write(OUT/'mean_metrics.csv',[mean])
(OUT/'verification.json').write_text(json.dumps(dict(status='passed',all_input_and_raw_SHA_checked=True,all_AUROC_recomputed=True,
    all_K50_image_scores_independently_recomputed=True,dense_nn_full_gallery_sample_checked=True,
    dense_nn_limit='3 spatial queries per category checked against float64 brute force; all 741 maps postprocessing checked, not all dense minima independently re-searched',audits=audits),indent=2))
env['status']='completed';(OUT/'environment.json').write_text(json.dumps(env,indent=2))
md=ROOT.parent/'markdown/SPADE_btad_execution.md';text=md.read_text(encoding='utf-8').split('\n## 자체 측정 결과')[0]
lines=['','## 자체 측정 결과','','제품 3개 비가중 평균, 단위 %. ImageNet 고정 모델이며 별도 학습과 test tuning은 없다.','',
    '| Product | Test images | Image AUROC | Pixel AUROC |','|---|---|---|---|']
for r in rows:lines.append(f"| {r['category']} | {r['n_images']} | {100*float(r['image_auroc']):.4f} | {100*float(r['pixel_auroc']):.4f} |")
lines += [f"| Mean | {mean['n_images']} | {100*mean['image_auroc']:.4f} | {100*mean['pixel_auroc']:.4f} |",'',
    '[검증](../source/results/btad_k50_seed42_20261008/verification.json) · [CSV](../source/results/btad_k50_seed42_20261008/category_metrics.csv)','']
md.write_text(text+'\n'.join(lines),encoding='utf-8')
subprocess.run([sys.executable,str(ROOT.parents[1]/'method10/source/build_musc_reproduction_tables.py')],check=True)
subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',r'C:\Users\test\Desktop\Codex\tools\update_obsidian_links.ps1'],check=True)
print('SPADE BTAD VERIFIED COMPLETE',flush=True)
