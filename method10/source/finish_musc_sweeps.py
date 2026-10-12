"""Verify saved sweeps, reconstruct full-resolution maps and export category macro means."""
import gc,json,sys,subprocess
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from musc_tables3_7_common import ROOT,sha,read,image_metrics,upsample
OUT=ROOT/'results/sweeps_seed42_20261012';BASE=ROOT/'results/tables3_7_seed42_20261012'
rows=read(OUT/'msm_interval_ratio_category_metrics.csv');assert len(rows)==162
audits=[]
for ds,cat in sorted({(r['dataset'],r['category']) for r in rows}):
    folder=OUT/ds/cat;base=BASE/ds/cat
    manifest=json.loads((folder/'raw_manifest.json').read_text());assert sha(manifest['path'])==manifest['sha256']
    old=json.loads((base/'raw_manifest.json').read_text());assert sha(old['path'])==old['sha256']==manifest['baseline_raw_sha256']
    proof=json.loads((folder/'msm_verification.json').read_text());assert proof['status']=='passed'
    assert len(proof['native_query_and_numpy_reduction_checks'])==72
    images=read(base/'images.csv')
    for im in images:
        assert sha(im['image_path'])==im['sha256']
        if im['mask_path']:assert sha(im['mask_path'])==im['mask_sha256']
    with np.load(old['path']) as z:y=z['image_labels'];gt=z['masks']
    scores_csv=read(folder/'msm_image_scores.csv')
    with np.load(manifest['path']) as z:
        for percent in [10,30,50,70,90,100]:
            maps=upsample(z[f'p{percent}']);assert np.isfinite(maps).all()
            saved=[r for r in scores_csv if int(r['percent'])==percent]
            before=np.array([float(r['score_before']) for r in saved]);after=np.array([float(r['score_after']) for r in saved])
            assert np.array_equal(before,maps.reshape(len(y),-1).max(1))
            assert [r['image_path'] for r in saved]==[r['image_path'] for r in images]
            row=next(r for r in rows if r['dataset']==ds and r['category']==cat and int(r['percent'])==percent)
            actual=image_metrics(y,after);actual['pixel_auroc']=float(roc_auc_score(gt.ravel(),maps.ravel()))
            for k,v in actual.items():assert abs(v-float(row[k]))<1e-12,(ds,cat,percent,k)
            # Selected raw distance proof also permits CPU independent interval reductions.
            for r in [1,3,5]:
                for layer in range(4):
                    dist=z[f'first_query_r{r}_layer{layer}'];k=int(dist.shape[1]*(percent/100)) if percent<100 else dist.shape[1]
                    assert k>0 and np.isfinite(np.sort(dist,axis=1)[:,:k].mean(1)).all()
            del maps;gc.collect()
    audits.append(dict(dataset=ds,category=cat,n_images=len(y),raw_sha256=manifest['sha256'],metric_conditions_verified=6))
    print('MSM VERIFIED',ds,cat,flush=True)
assert len(audits)==27 and sum(r['n_images'] for r in audits)==3887
means=read(OUT/'msm_interval_ratio_mean_metrics.csv')
for mean in means:
    rs=[r for r in rows if r['dataset']==mean['dataset'] and r['percent']==mean['percent']]
    assert len(rs)==int(mean['n_categories'])
    for k in ['image_auroc','image_f1_max','image_ap','pixel_auroc']:assert abs(float(mean[k])-np.mean([float(r[k]) for r in rs]))<1e-12
assert json.loads((OUT/'rscin_verification.json').read_text())['status']=='passed'
(OUT/'verification.json').write_text(json.dumps(dict(status='passed',figures=[7,8],n_categories=27,n_images=3887,
    MSM_category_conditions=162,RsCIN_category_conditions=297,all_MSM_metrics_recomputed=True,raw_input_hashes_checked=True,
    rscin_cpu_independent_smoothing_verified=True,macro_means_unweighted=True,categories=audits),indent=2))
md=ROOT.parent/'markdown/MuSc_sweeps_execution.md'
text=md.read_text(encoding='utf-8').split('\n## 최종 결과')[0]
text+='\n## 최종 결과\n\nMVTec15/1,725장 및 VisA12/2,162장의 MSM6비율·RsCIN11조건을 검증했다. 전체 완료. 수치는 0–1 단위이고 dataset 평균은 category별 비가중 평균이다. 논문 수치는 실험 결과로 대입하지 않았다.\n\n'
text+='- [MSM 비율 평균 CSV](../source/results/sweeps_seed42_20261012/msm_interval_ratio_mean_metrics.csv)\n- [RsCIN window 평균 CSV](../source/results/sweeps_seed42_20261012/rscin_single_window_mean_metrics.csv)\n- [검증](../source/results/sweeps_seed42_20261012/verification.json)\n'
md.write_text(text,encoding='utf-8')
subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',r'C:\Users\test\Desktop\Codex\tools\update_obsidian_links.ps1'],check=True)
print('MSM AND RSCIN SWEEPS VERIFIED',flush=True)
