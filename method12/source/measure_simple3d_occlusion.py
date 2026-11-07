"""Paired MiniShift capsule evaluation with the existing full GPU KD-tree backend."""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import compare_simple3d_worker as trace
import numpy as np
import psutil
import simple3d_gpu_kdtree as backend
import torch
from fpfh_only_publish import replace_with_retry
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'results/capsule_occlusion_gpu_kdtree_20261107'
ASSETS = Path('/home/test/simple3d-capsule-occlusion-20261107')
REPORT = HERE.parent / 'markdown/W46_Simple3D_capsule_occlusion_GPU_KDTree.md'
CASES = ['full', 'occluded_50', 'random_removed_50', 'occluded_75', 'random_removed_75']


def dump(path, value):
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    replace_with_retry(temp, path)


def visibility(xyz, i):
    assert xyz.shape == (500000, 3)
    n = len(xyz)
    # Keep the low-x side of a plane. Ties retain original index order.
    ranked = np.argsort(xyz[:, 0], kind='stable')
    random = np.random.default_rng(np.random.SeedSequence([20261107, i])).permutation(n)
    return dict(full=np.arange(n), occluded_50=np.sort(ranked[:n//2]),
        random_removed_50=np.sort(random[:n//2]), occluded_75=np.sort(ranked[:n//4]),
        random_removed_75=np.sort(random[:n//4]))


def score_metrics(gt, scores):
    if len(np.unique(gt)) < 2:
        return [None, None]
    return [float(roc_auc_score(gt, scores)), float(average_precision_score(gt, scores))]


def write_report(results, fit_s, elapsed):
    loc = '../source/results/capsule_occlusion_gpu_kdtree_20261107'
    lines = ['# W46 Simple3D — 캡슐 부분 가림·점 수 대조 실험', '',
        '- 기존 G: FPFH와 128개·12개 이웃 탐색을 GPU KD-tree로 실행함. 기존 모델 코드 수정 없음.',
        '- 원본 정상 54개로 메모리뱅크 한 번 구성. 동일 테스트 130개에 5조건 적용함.',
        '- 50% 가림은 25만 점, 75% 가림은 12만 5천 점을 남김. 낮은 X 쪽 연속 영역을 보존하는 평면 절단이며 실제 카메라 가림 시뮬레이션은 아님.',
        '- 무작위 대조군도 같은 점 수를 남김. 마스크는 변형 위치 선정에 사용하지 않음. 원본 전처리 뒤 잘라 동일 좌표계 유지.',
        '- 입력별 조건 실행 순서를 순환함. 모든 조건에서 동일 메모리뱅크, 원본 정답에서 살아남은 점의 라벨만 사용함.',
        '- 물체 AUROC는 원래 정상·불량 라벨 사용. 결함이 완전히 가려진 개수 별도 기록. 같은 점 비교는 원본 점수를 해당 조건과 동일한 점에 제한한 값임.',
        '- 평균 추론=특징 추출+점수 계산. 입력 읽기·가림 생성·진단·저장·지표 계산 제외. 세부 단계의 CUDA event와 wall time은 중첩되어 단순 합산할 수 없음.',
        f'- 뱅크 구성 {fit_s:.2f}초, 전체 실행 {elapsed/60:.2f}분.', '',
        '| 조건 | 남은 점 | 물체 AUROC (%) | 포인트 AUROC (%) | AP (%) | 같은 점의 원본 AP (%) | AP 변화 (%p) | 결함 미관측 | 평균 추론 (ms) | p95 (ms) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(x, scale=1):
        return '—' if x is None else f'{x*scale:.2f}'
    for r in results:
        lines.append(f"| {r['condition']} | {r['points']} | {fmt(r['object_auroc'],100)} | {fmt(r['point_auroc'],100)} | {fmt(r['point_ap'],100)} | {fmt(r['full_same_points_ap'],100)} | {fmt(r['paired_ap_delta'],100)} | {r['hidden_defects']} | {fmt(r['mean_ms'])} | {fmt(r['p95_ms'])} |")
    lines += ['', '## 동일 점 수에서 가림과 무작위 축소 비교', '',
        '가림−무작위 값은 같은 개수의 점을 남긴 두 조건의 차이임. AP는 두 조건이 서로 다른 점을 남겨 결함 비중이 다를 수 있으므로, 같은 점의 원본 대비 AP 변화도 함께 확인함.', '',
        '| 제거 비율 | 물체 AUROC 차이 (%p) | 포인트 AUROC 차이 (%p) | AP 차이 (%p) | 원본 대비 AP 변화 차이 (%p) | 추론 차이 (ms) |',
        '|---|---:|---:|---:|---:|---:|']
    keyed = {r['condition']:r for r in results}
    for n in (50,75):
        a,b=keyed[f'occluded_{n}'],keyed[f'random_removed_{n}']
        cells=[fmt(None if a[k] is None or b[k] is None else a[k]-b[k],s)
               for k,s in [('object_auroc',100),('point_auroc',100),('point_ap',100),('paired_ap_delta',100),('mean_ms',1)]]
        lines.append(f"| {n}% | {' | '.join(cells)} |")
    lines += ['', '## 단계별 추론 시간', '',
        '테스트 단계별 총 시간을 130샘플로 나눈 값(ms/샘플). 가림 생성·저장·평가·별도 이웃 진단 시간은 CSV에서 구분함.', '',
        '| 단계 | ' + ' | '.join(CASES) + ' |',
        '|---|' + '---:|'*len(CASES)]
    stages = defaultdict(dict)
    with (ROOT/'stage_summary.csv').open() as file:
        for r in csv.DictReader(file):
            if r['phase']=='test':
                stages[r['stage']+' / '+r['detail']][r['condition']]=float(r['ms_per_sample'])
    for k,v in sorted(stages.items()):
        lines.append('| '+k.replace('|','/')+' | '+' | '.join(fmt(v.get(c)) for c in CASES)+' |')
    lines += ['', '## 같은 위치에서 점수와 이웃 범위가 바뀌었는지', '',
        '표의 점수 차이는 변형 점수−원본 점수임. 각 샘플의 평균 차이를 구하고 샘플 간 중앙값을 취함. 결함 점수는 결함이 남은 샘플만 사용함. 가림 경계는 남은 점 중 절단면에 가장 가까운 X 상위 10%이며, 무작위 조건도 같은 방식의 X 상위 10%를 대조군으로 사용함. 이웃 거리는 각 조건에서 뽑은 동일한 점 256개의 원본·변형 비교임.', '',
        '| 조건 | 결함 점수 변화 | 정상 경계 점수 변화 | 경계 밖 정상 점수 변화 | 40번째 이웃 거리 배수 | 80번째 | 120번째 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    with (ROOT/'paired_diagnostics.csv').open() as file:
        diags=list(csv.DictReader(file))
    for c in CASES[1:]:
        selected=[r for r in diags if r['condition']==c]
        cells=[]
        for key in ('defect_score_change','normal_near_plane_score_change','normal_away_score_change'):
            a=[float(r[key]) for r in selected if r[key]]
            cells.append(fmt(float(np.median(a)) if a else None))
        for k in (40,80,120):
            a=[float(r[f'altered_k{k}_distance_median'])/float(r[f'original_k{k}_distance_median']) for r in selected if float(r[f'original_k{k}_distance_median'])>0]
            cells.append(fmt(float(np.median(a)) if a else None))
        lines.append(f"| {c} | {' | '.join(cells)} |")
    lines += ['', '## 자원 사용량', '',
        'GPU 사용률·전력은 테스트 구간 표본 평균, 전체 VRAM·CPU RSS는 표본 최대. Torch 할당은 샘플별 최대치 중 최댓값이며 CuPy 할당 제외. 전체 VRAM은 다른 작업과 화면 출력 포함.', '',
        '| 조건 | GPU 사용률 (%) | 전체 VRAM 최대 (MiB) | 전력 (W) | CPU RSS 최대 (GiB) | Torch 할당 최대 (GiB) |',
        '|---|---:|---:|---:|---:|---:|']
    with (ROOT/'resources.csv').open() as file:
        resources=list(csv.DictReader(file))
    with (ROOT/'test_timings.csv').open() as file:
        timings=list(csv.DictReader(file))
    for c in CASES:
        q=[r for r in resources if r['phase']=='test' and r['condition']==c]
        def stat(key,fn,divisor=1):
            a=[float(r[key])/divisor for r in q if r[key] not in ('','[N/A]','N/A','[Not Supported]')]
            return fmt(float(fn(a)) if a else None)
        peak=max(int(r['torch_peak_bytes']) for r in timings if r['condition']==c)/2**30
        lines.append(f"| {c} | {stat('gpu_util_pct',np.mean)} | {stat('vram_mib',max)} | {stat('power_w',np.mean)} | {stat('cpu_rss_bytes',max,2**30)} | {fmt(peak)} |")
    lines += ['', '## 원본과 원인 확인 근거', '',
        '- 아래 기록에서 남은 결함 점, 같은 위치 점수 변화, 정상 절단 경계 주변의 점수 상승, 같은 위치에서 40·80·120번째 이웃 거리 변화를 함께 확인함. 한 카테고리·한 절단 방향 결과로 일반화하지 않음.',
        f'- [최종 지표]({loc}/metrics.csv), [샘플별 추론과 결함 보존]({loc}/test_timings.csv), [단계별 요약]({loc}/stage_summary.csv), [단계 호출 원본]({loc}/stage_timings.csv).',
        f'- [같은 점의 점수·이웃 거리 진단]({loc}/paired_diagnostics.csv), [자원 시계열]({loc}/resources.csv), [실행 검증]({loc}/progress.json).',
        '- [실행 코드](../source/measure_simple3d_occlusion.py), [실행 스크립트](../source/run_simple3d_occlusion.sh).']
    REPORT.write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main(smoke):
    out = ROOT/'preflight' if smoke else ROOT
    asset = ASSETS/'preflight' if smoke else ASSETS
    out.mkdir(parents=True,exist_ok=True); asset.mkdir(parents=True,exist_ok=True)
    assert not (out/'test_timings.csv').exists(), 'Existing measurement preserved'
    state = {}
    def progress(phase, index=-1, condition='-', **kw):
        trace.CONTEXT.update(phase=phase,index=index,condition=condition)
        state.update(status='running',phase=phase,index=index,condition=condition,heartbeat_unix=time.time(),**kw)
        dump(out/'progress.json',state)
    obs=trace.observed
    stop=threading.Event()
    start=time.perf_counter()
    def monitor():
        proc=psutil.Process()
        while not stop.is_set():
            try:
                snap=state.copy()
                raw=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu','--format=csv,noheader,nounits'],text=True,timeout=4).splitlines()[0].split(',')
                rss=0
                for p in [proc]+proc.children(recursive=True):
                    try:rss+=p.memory_info().rss
                    except psutil.Error:pass
                trace.append(out/'resources.csv',[dict(elapsed_s=time.perf_counter()-start,phase=snap.get('phase'),index=snap.get('index'),condition=snap.get('condition'),gpu_util_pct=raw[0].strip(),vram_mib=raw[1].strip(),power_w=raw[2].strip(),temperature_c=raw[3].strip(),cpu_rss_bytes=rss)])
            except (OSError,subprocess.SubprocessError,IndexError):
                pass
            stop.wait(.5)
    progress('startup',pid=os.getpid(),samples_completed=0,target_samples=2 if smoke else 130)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        args=SimpleNamespace(dataset='minishift',device='cuda:0',num_group=4096,group_size=128,
            max_nn=40,use_LFSA=True,use_MSND=True,num_MSND=2,feature='FPFH',level='ALL',vis_save=False)
        model=trace.PatchCore(args=args)
        train=model.get_dataloader('minishift','train','capsule','ALL')
        test=model.get_dataloader('minishift','test','capsule','ALL')
        assert len(train)==54 and len(test)==130
        old=HERE/'results/official_vs_gpu_kdtree_20261104/gpu_kdtree/minishift/capsule'
        reference=json.loads((old/'manifest.json').read_text())
        manifest=dict(train=train.dataset.pcd_paths,test=[dict(path=p,label=int(y)) for p,y in zip(test.dataset.pcd_paths,test.dataset.labels)])
        assert manifest==reference, 'Input list changed'
        dump(out/'manifest.json',manifest)
        baseline_env=json.loads((old/'environment.json').read_text())
        assert {str(p.relative_to(trace.REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in trace.REPO.rglob('*.py')}==baseline_env['source_sha256']
        dump(out/'environment.json',dict(args=vars(args),torch=torch.__version__,gpu=torch.cuda.get_device_name(),
            baseline=baseline_env,seed=20261107,conditions=CASES,
            sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                     [Path(__file__),HERE/'compare_simple3d_worker.py',HERE/'simple3d_gpu_kdtree.py',HERE/'simple3d_gpu_kdtree.cu']}))
        trace.instrument();backend.install(trace,detail_tracing=True)
        samples=[]; all_rows=defaultdict(list)
        with torch.inference_mode():
            progress('warmup')
            model.method.get_features(next(iter(train))[0]);trace.flush(out)
            fit_start=time.perf_counter()
            for i,(pc,_,_,_) in enumerate(train):
                progress('fit',i)
                obs('collect_features_total',lambda:model.method.collect_features(pc),'cpu_gpu_wall','unchanged G');trace.flush(out)
                if smoke:break
            progress('coreset')
            obs('coreset_total',model.method.run_coreset,'cpu_gpu_wall','unchanged 5% coreset');trace.flush(out)
            fit_s=time.perf_counter()-fit_start
            torch.save(model.method.patch_lib.detach().cpu(),asset/'memory_bank.pt')
            iterator=iter(test)
            for i in range(2 if smoke else len(test)):
                progress('prepare',i)
                pc,mask,label,path=obs('input_loading',lambda:next(iterator),'cpu_wall','official loader')
                xyz=pc[0].numpy()
                gt_full=mask.numpy().reshape(-1).astype(np.uint8)
                keep=obs('visibility_index_build',lambda:visibility(xyz,i),'cpu_wall','deterministic paired conditions')
                trace.flush(out)
                preds={}
                for c in CASES[i%5:]+CASES[:i%5]:
                    idx=keep[c]
                    progress('prepare',i,c)
                    points,gt=obs('apply_visibility',lambda:(pc[:,idx,:].contiguous(),mask[...,idx].contiguous()),'cpu_wall','preserve index mapping')
                    progress('test',i,c)
                    model.method.init_para();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    t=time.perf_counter()
                    feat,raw,dense,centers=model.method.get_features(points);torch.cuda.synchronize()
                    f=time.perf_counter()-t;t=time.perf_counter()
                    model.method.compute_anomay_scores(feat,gt,label,path,raw,dense,centers);torch.cuda.synchronize()
                    s=time.perf_counter()-t
                    pred=np.asarray(model.method.predictions[-1],dtype=np.float32).reshape(-1).copy()
                    assert np.isfinite(pred).all() and len(pred)==len(idx)
                    preds[c]=pred
                    row=dict(index=i,condition=c,path=path[0],label=int(label.item()),points=len(idx),original_defect_points=int(gt_full.sum()),visible_defect_points=int(gt_full[idx].sum()),object_score=float(model.method.image_preds[-1]),feature_s=f,score_s=s,inference_s=f+s,torch_peak_bytes=torch.cuda.max_memory_allocated())
                    all_rows[c].append(row);trace.append(out/'test_timings.csv',[row])
                    progress('save',i,c)
                    obs('point_scores_save',lambda:np.savez_compressed(asset/f'{i:03d}_{c}.npz',scores=pred,indices=idx.astype(np.int32),gt=gt_full[idx]),'cpu_wall','scores and original point IDs')
                    trace.flush(out)
                    del feat,raw,dense,centers;model.method.init_para()
                progress('diagnostic',i)
                tree=cKDTree(xyz)
                for c in CASES[1:]:
                    idx=keep[c];p=preds[c];g=gt_full[idx];change=p-preds['full'][idx]
                    near=xyz[idx,0]>=np.quantile(xyz[idx,0],.9)
                    mean=lambda a:float(a.mean()) if a.size else None
                    q=np.random.default_rng(np.random.SeedSequence([20261108,i])).choice(idx,256,replace=False)
                    original=tree.query(xyz[q],k=[40,80,120])[0]
                    altered=cKDTree(xyz[idx]).query(xyz[q],k=[40,80,120])[0]
                    diag=dict(index=i,condition=c,visible_defect_points=int(g.sum()),normal_score_change=mean(change[g==0]),defect_score_change=mean(change[g==1]),normal_near_plane_score_change=mean(change[near&(g==0)]),normal_away_score_change=mean(change[~near&(g==0)]))
                    for j,k in enumerate([40,80,120]):
                        diag[f'original_k{k}_distance_median']=float(np.median(original[:,j]))
                        diag[f'altered_k{k}_distance_median']=float(np.median(altered[:,j]))
                    trace.append(out/'paired_diagnostics.csv',[diag])
                samples.append(i)
                progress('test_sample_complete',i,samples_completed=len(samples))
        if not smoke:
            results=[]
            for c in CASES:
                progress('metrics',condition=c)
                ss=[]; gg=[]; refs=[]
                for i in samples:
                    with np.load(asset/f'{i:03d}_{c}.npz') as z:
                        ss.append(z['scores']);gg.append(z['gt'])
                        with np.load(asset/f'{i:03d}_full.npz') as original:refs.append(original['scores'][z['indices']])
                p,g,r=np.concatenate(ss),np.concatenate(gg),np.concatenate(refs)
                au,ap=score_metrics(g,p);rau,rap=score_metrics(g,r)
                rows=all_rows[c]
                labels=[x['label'] for x in rows];scores=[x['object_score'] for x in rows]
                times=np.array([x['inference_s'] for x in rows])*1000
                visible=[j for j,x in enumerate(rows) if not x['label'] or x['visible_defect_points']>0]
                va,_=score_metrics(np.array(labels)[visible],np.array(scores)[visible])
                results.append(dict(condition=c,points=rows[0]['points'],object_auroc=float(roc_auc_score(labels,scores)),visible_object_auroc=va,point_auroc=au,point_ap=ap,full_same_points_auroc=rau,full_same_points_ap=rap,paired_ap_delta=None if ap is None or rap is None else ap-rap,hidden_defects=sum(x['label'] and not x['visible_defect_points'] for x in rows),visible_defect_points=sum(x['visible_defect_points'] for x in rows),mean_ms=float(times.mean()),p95_ms=float(np.percentile(times,95))))
                del ss,gg,refs,p,g,r
            trace.append(out/'metrics.csv',results)
            group=defaultdict(list)
            with (out/'stage_timings.csv').open() as file:
                for x in csv.DictReader(file):
                    detail=x['detail']
                    # Normalize point-count-specific labels so stage columns align.
                    stage=x['stage']
                    if stage=='coordinate_knn_gpu_kdtree':detail=detail.split(';')[0]
                    elif stage.startswith('fpfh_gpu_kdtree_scale_'):pass
                    else:detail=x['device']
                    group[x['phase'],x['condition'],stage,detail].append(float(x['seconds']))
            trace.append(out/'stage_summary.csv',[dict(phase=k[0],condition=k[1],stage=k[2],detail=k[3],calls=len(v),total_s=sum(v),mean_call_ms=np.mean(v)*1000,ms_per_sample=sum(v)*1000/len(samples)) for k,v in sorted(group.items())])
            assert len(samples)==130 and all(len(v)==130 for v in all_rows.values())
            write_report(results,fit_s,time.perf_counter()-start)
        state.update(status='passed_smoke' if smoke else 'complete_verified',heartbeat_unix=time.time())
        dump(out/'progress.json',state)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc),heartbeat_unix=time.time())
        dump(out/'progress.json',state)
        raise
    finally:
        stop.set();thread.join()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    main(parser.parse_args().smoke)
