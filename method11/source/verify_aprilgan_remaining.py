"""Final evidence audit: all seeds, references, metric means, raw caches, RsCIN."""
import csv
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
RAW=Path('/home/test/aprilgan_results')
KEYS=['image_auroc','image_f1_max','image_ap','pixel_auroc','pixel_f1_max','pixel_ap','aupro']

def read(p):
    with p.open(encoding='utf-8-sig') as f: return list(csv.DictReader(f))

cases=[('mvtec','zero_shot',42,'mvtec_zero_shot_20261006')]
for ds in ['mvtec','visa','btad']:
    if ds!='mvtec': cases.append((ds,'zero_shot',42,f'{ds}_0shot_seed42_20261006'))
    cases.extend((ds,'few_shot',s,f'{ds}_4shot_seed{s}_20261006') for s in [42,43,44])
small_tag='mvtec_0shot_vit_b16_plus_240_20261006'
cases.append(('mvtec','zero_shot',42,small_tag))
dataset_paths={'mvtec':Path('/home/test/data/mvtec'),'visa':Path('/home/test/data/VisA_20220922'),
               'btad':Path('/home/test/data/btad_original/BTech_Dataset_transformed')}
expected={'mvtec':(15,1725),'visa':(12,2162),'btad':(3,741)}
baselines={};details=[];means={}
for ds,mode,seed,tag in cases:
    path=ROOT/'results'/tag
    env=json.loads((path/'environment.json').read_text())
    assert env['status']=='completed',tag
    rows=read(path/'category_metrics.csv')
    assert len(rows)==expected[ds][0] and sum(int(r['n_images']) for r in rows)==expected[ds][1]
    mean=read(path/'mean_metrics.csv')[0]
    assert int(mean['n_images'])==expected[ds][1]
    for k in KEYS:
        numbers=[float(r[k]) for r in rows]
        # sklearn AP may return 1.0000000000000002 for perfect ranking.
        # Preserve raw measurements; tolerate only floating-point roundoff.
        assert all(math.isfinite(x) and -1e-12<=x<=1+1e-12 for x in numbers),(tag,k)
        assert abs(statistics.mean(numbers)-float(mean[k]))<1e-12,(tag,k)
    means[tag]=mean
    images={}
    for row in rows:
        c=row['category']
        scores=read(path/c/'image_scores.csv')
        assert len(scores)==int(row['n_images'])
        images.update({r['image_path']:(r['label'],r['sha256']) for r in scores})
        assert (RAW/tag/c/'raw_predictions.npz').is_file()
    if ds in baselines: assert images==baselines[ds],tag
    else: baselines[ds]=images
    assert len(images)==expected[ds][1],tag
    if tag!='mvtec_zero_shot_20261006':
        compact=json.loads((path/'duplicate_cache_verification.json').read_text())
        assert compact['status']=='passed' and compact['n_images']==expected[ds][1]
    if mode=='few_shot':
        refs=json.loads((path/'references.json').read_text())
        meta=json.loads((dataset_paths[ds]/'meta.json').read_text())
        allowed={r['img_path'] for values in meta['train'].values() for r in values if not r['anomaly']}
        assert len(refs)==4*expected[ds][0]
        reference_checksums=read(path/'reference_checksums.csv')
        assert len(reference_checksums)==len(refs) and all(r['label']=='0' and len(r['sha256'])==64 for r in reference_checksums)
        assert all(r['img_path'] in allowed and not r['anomaly'] for r in refs)
        for category in meta['test']: assert sum(r['cls_name']==category for r in refs)==4
    if ds!='btad' and tag!=small_tag:
        rp=path/'rscin'
        assert json.loads((rp/'provenance.json').read_text())['status']=='completed'
        rmean=read(rp/'mean_metrics.csv')
        before=next(r for r in rmean if r['rscin']=='w/o')
        for k in KEYS[:3]: assert abs(float(before[k])-float(mean[k]))<1e-12
    details.append(dict(tag=tag,n_categories=len(rows),n_images=len(images),status='verified'))
aggregates=read(ROOT/'results/aprilgan_remaining_20261006/aggregates.csv')
assert len(aggregates)==6
for row in aggregates:
    tags=row['source_tags'].split(';')
    assert int(row['n_runs'])==(3 if row['mode']=='few_shot' else 1)
    for k in KEYS:
        numbers=[float(means[tag][k]) for tag in tags]
        assert abs(statistics.mean(numbers)-float(row[k]))<1e-12
        if len(numbers)>1: assert abs(statistics.stdev(numbers)-float(row[k+'_std']))<1e-12
for backbone in ['ViT-L-14-336','ViT-B-16-plus-240']:
    bench=json.loads((ROOT/'results'/f'benchmark_{backbone}_20261006/benchmark.json').read_text())
    assert bench['status']=='completed' and bench['n_images']==1725 and bench['mean_ms']>0
training=json.loads((ROOT/'results/train_vit_b16_plus_240_20261006/environment.json').read_text())
assert training['status']=='completed' and training['checkpoint_all_tensors_finite']
result=dict(status='passed',run_count=len(cases),records=details,
            image_sets_identical_across_seeds=True,all_reference_images_from_normal_train=True,
            six_aggregate_rows_verified=True,metrics_and_std_verified=True,
            isolated_benchmarks_verified=True,small_backbone_training_verified=True,
            duplicate_cache_exact_comparisons_verified=True,
            paper_protocol_identity_certified=False)
result['metric_boundary_tolerance']=1e-12
result['raw_metrics_clipped']=False
(ROOT/'results/aprilgan_remaining_20261006/final_verification.json').write_text(json.dumps(result,indent=2))
print('FINAL VERIFICATION PASSED',len(cases),'runs; 6 aggregates; references; RsCIN; 2 benchmarks; trained checkpoint.')
