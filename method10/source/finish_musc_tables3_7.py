"""Independently reconstruct and verify all table values from preserved patch predictions."""
import gc,json,os,subprocess,sys
from collections import defaultdict
import numpy as np
from musc_tables3_7_common import *
os.chdir(REPO);sys.path.insert(0,str(REPO))
from models.modules._RsCIN import RsCIN
rows=read(OUT/'category_metrics.csv');groups=defaultdict(list);audits=[]
for row in rows:groups[(row['dataset'],row['category'])].append(row)
assert len(groups)==27
counts={'mvtec_ad':0,'visa':0}
for (ds,cat),catrows in groups.items():
    assert len(catrows)==17 and len({(r['table'],r['condition']) for r in catrows})==17
    folder=OUT/ds/cat;manifest=json.loads((folder/'raw_manifest.json').read_text())
    assert sha(manifest['path'])==manifest['sha256']
    images=read(folder/'images.csv');n=len(images);counts[ds]+=n
    for im in images:
        assert sha(im['image_path'])==im['sha256']
        if im['mask_path']:assert sha(im['mask_path'])==im['mask_sha256']
    with np.load(manifest['path']) as z:
        y=z['image_labels'];cls=z['class_tokens'];gt=z['masks']
        assert len(y)==n and gt.shape==(n,518,518) and np.array_equal(y,[int(r['label']) for r in images])
        cache={};score_rows=read(folder/'image_scores.csv')
        for row in catrows:
            table=int(row['table']);condition=row['condition'];patches=patch_for(z,table,condition)
            key=hashlib.sha256(patches.tobytes()).hexdigest()
            if key not in cache:
                maps=upsample(patches);assert np.isfinite(maps).all()
                before=maps.reshape(n,-1).max(1);after=RsCIN(before,cls,k_list=[1,2,3] if ds=='mvtec_ad' else [1,8,9])
                # Independent CPU matrix implementation of native neighborhood smoothing.
                similarity=(cls@cls.T).astype(np.float32)
                normalized=((before-before.min())/(before.max()-before.min())).astype(np.float32)
                smooth=[]
                for k in ([1,2,3] if ds=='mvtec_ad' else [1,8,9]):
                    ids=torch.topk(torch.from_numpy(similarity),n-k,largest=False,sorted=True).indices.numpy()
                    w=similarity.copy();w[np.arange(n)[:,None],ids]=0
                    w=w/w.sum(1,keepdims=True);smooth.append(w@normalized)
                independent=np.mean(np.stack(smooth,1),1)
                assert np.allclose(after,independent,rtol=1e-5,atol=1e-6)
                pixel=float(roc_auc_score(gt.ravel(),maps.ravel()))
                cache[key]=(before,after,pixel);del maps;gc.collect()
            before,after,pixel=cache[key];scores=before if table==5 and condition=='w/o' else after
            actual=image_metrics(y,scores)
            if table!=5:actual['pixel_auroc']=pixel
            for k,v in actual.items():assert abs(v-float(row[k]))<1e-12,(ds,cat,condition,k)
            stored=[r for r in score_rows if int(r['table'])==table and r['condition']==condition]
            assert [r['image_path'] for r in stored]==[r['image_path'] for r in images]
            assert np.array_equal(scores,[float(r['score_used']) for r in stored])
        membership=read(folder/'subset_membership.csv')
        for s in [2,3]:
            sub=[r for r in membership if int(r['s'])==s]
            assert sorted(int(r['full_index']) for r in sub)==list(range(n))
            for r in sub:assert r['image_path']==images[int(r['full_index'])]['image_path']
    audits.append(dict(dataset=ds,category=cat,n_images=n,conditions=17,raw_sha256=manifest['sha256'],all_metrics_recomputed=True))
    print('VERIFIED',ds,cat,flush=True);gc.collect()
assert counts=={'mvtec_ad':1725,'visa':2162}
aggregate=defaultdict(list)
for r in rows:aggregate[(r['table'],r['dataset'],r['condition'])].append(r)
means=[]
for (table,ds,condition),rs in aggregate.items():
    assert len(rs)==(15 if ds=='mvtec_ad' else 12)
    mean=dict(table=int(table),dataset=ds,condition=condition,n_categories=len(rs),n_images=sum(int(r['n_images']) for r in rs))
    for k in ['image_auroc','image_f1_max','image_ap','pixel_auroc']:
        mean[k]=float(np.mean([float(r[k]) for r in rs]))
    mean.update(subset_min=min(int(r['subset_min']) for r in rs),subset_max=max(int(r['subset_max']) for r in rs))
    means.append(mean)
write(OUT/'mean_metrics.csv',means)
bench=json.loads((OUT/'benchmark_verification.json').read_text());assert bench['status']=='passed'
timings=read(OUT/'timing_samples.csv');assert len(timings)==9 and sha(OUT/'timing_samples.csv')==bench['timing_samples_sha256']
for r in timings:assert float(r['total_ms_per_image'])>=float(r['core_ms_per_image'])>0
(OUT/'verification.json').write_text(json.dumps(dict(status='passed',n_categories=27,n_images=3887,n_category_conditions=len(rows),
    datasets=counts,tables=[3,4,5,6,7],all_metrics_recomputed=True,raw_hashes_checked=True,input_hashes_checked=True,
    macro_means_unweighted=True,benchmark=bench,categories=audits),indent=2))
md=ROOT.parent/'markdown/MuSc_tables3_7_execution.md'
text=md.read_text(encoding='utf-8').split('\n## 최종 결과')[0]
text+='\n## 최종 결과\n\n27개 category/3,887장의 459개 category 조건과 속도9개 반복 측정을 검증했다. Table3/4/5/7 수치는 category별 비가중 평균이며 Table6는 RTX5080 VisA candle200 pool이다. 논문 수치를 실험값으로 대입하지 않았다.\n\n'
text+='[평균 CSV](../source/results/'+TAG+'/mean_metrics.csv) · [검증](../source/results/'+TAG+'/verification.json) · [속도 CSV](../source/results/'+TAG+'/benchmark_summary.csv)\n'
md.write_text(text,encoding='utf-8')
subprocess.run([sys.executable,str(ROOT/'build_musc_reproduction_tables.py')],check=True)
subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',r'C:\Users\test\Desktop\Codex\tools\update_obsidian_links.ps1'],check=True)
print('MUSC TABLES3-7 VERIFIED',flush=True)
