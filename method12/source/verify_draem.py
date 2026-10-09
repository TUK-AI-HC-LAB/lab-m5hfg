"""Recompute native metrics from preserved DRAEM raw predictions and audit data."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score,roc_auc_score

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/mvtec_public_20261007'
RAW=Path('/home/test/draem_results/mvtec_public_20261007')
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
env=json.loads((OUT/'environment.json').read_text())
assert env['status']=='completed'
assert len(env['checkpoint_hashes'])==30
metrics=read(OUT/'category_metrics.csv')
assert len(metrics)==15 and sum(int(r['n_images']) for r in metrics)==1725
metric_keys=['image_auroc','image_f1_max','image_ap','pixel_auroc','pixel_f1_max','pixel_ap','aupro']
assert all(np.isfinite(float(r[k])) and -1e-12<=float(r[k])<=1+1e-12 for r in metrics for k in metric_keys)
audit=[]
for row in metrics:
    category=row['category'];images=read(OUT/category/'image_scores.csv')
    manifest=json.loads((OUT/category/'raw_manifest.json').read_text())
    path=Path(manifest['path'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['sha256']
    with np.load(path) as z:
        labels,scores,maps,masks=[z[k] for k in ['image_labels','image_scores','anomaly_maps','masks']]
    assert len(images)==int(row['n_images'])==len(labels)
    assert maps.shape==masks.shape==(len(images),256,256)
    assert np.isfinite(maps).all() and np.isfinite(scores).all()
    assert maps.min()>=0 and maps.max()<=1 and set(np.unique(masks))=={0,1}
    for i,r in enumerate(images):
        assert int(r['label'])==int(labels[i]) and float(r['score'])==float(scores[i])
        assert hashlib.sha256(Path(r['image_path']).read_bytes()).hexdigest()==r['sha256']
    # Raw values originated as float32; float64 arrays are native test buffers.
    # Reapply official pooling on the GPU to verify image-score construction.
    with torch.no_grad():
        derived=[]
        for m in maps:
            x=torch.from_numpy(m.astype(np.float32))[None,None].cuda()
            derived.append(float(torch.nn.functional.avg_pool2d(x,21,stride=1,padding=10).max().cpu()))
    assert np.array_equal(np.asarray(derived,dtype=np.float32),scores.astype(np.float32))
    check=dict(image_auroc=roc_auc_score(labels,scores),image_ap=average_precision_score(labels,scores),
               pixel_auroc=roc_auc_score(masks.ravel(),maps.ravel()),pixel_ap=average_precision_score(masks.ravel(),maps.ravel()))
    errors={k:abs(float(v)-float(row[k])) for k,v in check.items()}
    assert max(errors.values())<1e-12
    audit.append(dict(category=category,n_images=len(images),raw_sha256_verified=True,
        image_score_pooling_exact=True,native_metrics_max_absolute_error=max(errors.values()),
        current_images_sha256_verified=True))
    print('VERIFIED',category,flush=True)
mean=read(OUT/'mean_metrics.csv')[0]
for key in metric_keys:
    assert abs(float(mean[key])-np.mean([float(r[key]) for r in metrics]))<1e-12
alignment=read(OUT/'rscin/dataset_alignment.csv')
assert len(alignment)==15 and sum(int(r['n_images']) for r in alignment)==1725
assert all(r['labels_and_sha256_match']=='True' for r in alignment)
result=dict(status='passed',n_categories=15,n_images=1725,n_checkpoints=30,native_metrics_recomputed=True,
    raw_predictions_verified=True,macro_means_verified=True,rscin_shared_feature_alignment_verified=True,
    categories=audit)
(OUT/'verification.json').write_text(json.dumps(result,indent=2))
print('VERIFICATION PASSED',flush=True)
