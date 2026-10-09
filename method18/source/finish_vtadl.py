"""Independent verification of retained BTAD raw maps and score components."""
import json,sys,subprocess
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from vtadl_common import *
env=json.loads((OUT/'environment.json').read_text());assert env['status']=='evaluated'
rows=read(OUT/'category_metrics.csv');assert len(rows)==3 and {r['category'] for r in rows}=={'01','02','03'}
reference=json.loads((DATA/'meta.json').read_text())
dataset_proof=json.loads((ROOT.parents[1]/'method11/source/results/btad_metadata_verification.json').read_text())
assert dataset_proof['status']=='passed' and sha(DATA/'meta.json')==dataset_proof['metadata_sha256']
audits=[]
for row in rows:
    cat=row['category'];folder=OUT/cat
    manifest=json.loads((folder/'raw_manifest.json').read_text())
    assert sha(manifest['path'])==manifest['sha256'] and sha(manifest['checkpoint'])==manifest['checkpoint_sha256']
    with np.load(manifest['path']) as z:
        y,s,maps,masks=[z[k] for k in ['image_labels','image_scores','anomaly_maps','masks']]
        mse,ssim,maximum,patch=[z[k] for k in ['mse','ssim','max_patch_density','patch_density']]
    images=read(folder/'test_images.csv');train=read(folder/'train_images.csv');scores=read(folder/'image_scores.csv')
    for phase,manifest_rows in [('train',train),('test',images)]:
        expected=reference[phase][cat]
        assert len(manifest_rows)==len(expected)
        for r,e in zip(manifest_rows,expected):
            assert str(Path(r['image_path']).relative_to(DATA))==e['img_path'] and int(r['label'])==int(e['anomaly'])
            assert (str(Path(r['mask_path']).relative_to(DATA)) if r['mask_path'] else '')==e['mask_path']
    assert len(images)==len(y)==int(row['n_images']) and maps.shape==masks.shape==(len(y),512,512)
    assert set(r['image_path'] for r in train).isdisjoint(r['image_path'] for r in images)
    for image,score,label,value in zip(images,scores,y,s):
        assert sha(image['image_path'])==image['sha256'] and int(image['label'])==int(label)==int(score['label']) and float(score['score'])==float(value)
        if image['mask_path']:assert sha(image['mask_path'])==image['mask_sha256']
    for image in train:assert sha(image['image_path'])==image['sha256'] and int(image['label'])==0
    assert np.allclose(s,mse+ssim+maximum,rtol=1e-6,atol=1e-4)
    assert np.array_equal(maximum,patch.max(1))
    # Recreate first native patch interpolation+filter independently on CPU.
    up=torch.nn.functional.interpolate(torch.tensor(patch[0]).reshape(1,1,8,8),size=(512,512),mode='bilinear',align_corners=True)
    rebuilt=gaussian_filter(up.numpy(),sigma=4)[0,0]
    assert np.allclose(rebuilt,maps[0],rtol=1e-6,atol=1e-5)
    recomputed=metrics(y,s,masks,maps)
    for k,v in recomputed.items():assert abs(v-float(row[k]))<1e-12
    history=read(folder/'training.csv');assert len(history)==400 and int(history[-1]['epoch'])==400
    minimum=min(float(r['mean_loss']) for r in history)
    assert int(row['checkpoint_epoch'])==max(int(r['epoch']) for r in history if float(r['mean_loss'])==minimum)
    proof=json.loads((folder/'first_gpu_step.json').read_text());assert proof['status']=='passed' and proof['finite_gradients']
    audits.append(dict(category=cat,n_train=len(train),n_test=len(images),epochs=400,checkpoint_epoch=int(row['checkpoint_epoch']),raw_sha256=manifest['sha256']))
means=dict(dataset='btad',n_categories=3,n_images=sum(int(r['n_images']) for r in rows),
    image_auroc=float(np.mean([float(r['image_auroc']) for r in rows])),pixel_auroc=float(np.mean([float(r['pixel_auroc']) for r in rows])))
assert means['n_images']==dataset_proof['test_images']==741
write(OUT/'mean_metrics.csv',[means])
(OUT/'verification.json').write_text(json.dumps(dict(status='passed',all_category_AUROCs_independently_recomputed=True,
    raw_hashes_and_all_input_hashes_verified=True,first_image_each_category_map_recomputed=True,image_score_formula_recomputed=True,audits=audits),indent=2))
env['status']='completed';(OUT/'environment.json').write_text(json.dumps(env,indent=2))
md=ROOT.parent/'markdown/VT_ADL_btad_execution.md'
text=md.read_text(encoding='utf-8').split('\n## 자체 측정 결과')[0]
lines=['','## 자체 측정 결과','','단위는 %. Mean은 제품3개 비가중 평균, seed123 단일 재학습이다.','',
    '| Product | Test images | Image AUROC | Pixel AUROC | Selected epoch |','|---|---|---|---|---|']
for r in rows:lines.append(f"| {r['category']} | {r['n_images']} | {100*float(r['image_auroc']):.4f} | {100*float(r['pixel_auroc']):.4f} | {r['checkpoint_epoch']} |")
lines+= [f"| Mean | {means['n_images']} | {100*means['image_auroc']:.4f} | {100*means['pixel_auroc']:.4f} | — |",'',
    '[검증](../source/results/btad_seed123_20261008/verification.json) · [카테고리 CSV](../source/results/btad_seed123_20261008/category_metrics.csv)','']
md.write_text(text+'\n'.join(lines),encoding='utf-8')
subprocess.run([sys.executable,str(ROOT.parents[1]/'method10/source/build_musc_reproduction_tables.py')],check=True)
subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',r'C:\Users\test\Desktop\Codex\tools\update_obsidian_links.ps1'],check=True)
print('VTADL BTAD TRAINING EVALUATION VERIFIED',flush=True)
