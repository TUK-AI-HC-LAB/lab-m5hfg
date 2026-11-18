"""One-variable exact-search pilot, using the SAME prebuilt STEP index."""
import hashlib
import statistics
import time
from pathlib import Path
import cupy as cp
import numpy as np
import torch
import run_simple3d_capsule_flashfps as bench
import measure_simple3d_capsule_step as step
from analyze_simple3d_post_top3 import ROOT, REPORT, analyze

OUT=ROOT/'pilot'


def kernel():
    # Compile in memory. Preserve the baseline source and every other kernel operation.
    original=(bench.HERE/'simple3d_step_index.cu').read_text()
    before='''    sort_pairs(bound,order,p);
    for(int c=0;c<p;++c) {
        if(bound[c]>best[119]) break; // strict, to include distance ties
        int g=order[c];'''
    after='''    // Seed with the minimum-bound cluster; do not sort all p bounds.
    if(t==0) {
        int first=0;
        for(int c=1;c<p;++c)
            if(after(bound[first],first,bound[c],c)) first=c;
        needed=first;
    }
    __syncthreads();
    int first=needed; // local copy before needed is reused for candidate merges
    __syncthreads();
    for(int c=-1;c<p;++c) {
        if(c==first) continue;
        int g=c<0?first:c;
        if(bound[g]>best[119]) continue; // no early break without sorted bounds'''
    assert original.count(before)==1
    changed=original.replace(before,after)
    return cp.RawKernel(changed,'search',options=('--std=c++17','--fmad=false')),hashlib.sha256(changed.encode()).hexdigest()


def timed(fn):
    cp.cuda.get_current_stream().synchronize();torch.cuda.synchronize()
    start=time.perf_counter();out=fn()
    cp.cuda.get_current_stream().synchronize();torch.cuda.synchronize()
    return out,1000*(time.perf_counter()-start)


def search(fn,x,index,p):
    out=cp.empty((len(x),120),cp.int32)
    fn((len(x),),(128,),(x,*index,np.int32(p),np.int32(len(x)),out))
    return out


def main():
    if (OUT/'plan.json').exists(): raise RuntimeError('Existing pilot preserved')
    analyze(); OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    fn,digest=kernel()
    bench.atomic(OUT/'plan.json',dict(question='Can nearest-bound seeding replace full cluster-bound sorting?',
        hypothesis='Same exact 120-NN and FPFH with less query work; extra visited clusters may refute speed hypothesis',
        fixed='Same input and index, STEP512/Lloyd50, all other distance and tie rules unchanged',
        targets={'minishift/capsule':[0,50,129],'shapenet/vase9':[0,18,36]},repeats=4,
        source_sha256={str(p):bench.sha(p) for p in (Path(__file__),bench.HERE/'simple3d_step_index.cu',bench.HERE/'measure_simple3d_capsule_step.py')},
        compiled_trial_sha256=digest,gpu=torch.cuda.get_device_name()))
    bench.atomic(OUT/'progress.json',dict(status='smoke',completed_samples=0))
    # Independent exhaustive check, including coordinate duplicates and distance ties.
    for tied in (False,True):
        x=cp.random.RandomState(81).random_sample((1024,3),dtype=cp.float32)
        if tied: x[1:8]=x[0]
        idx=step.indexing(x,64,5)
        a=step.step_search(x,idx,64); b=search(fn,x,idx,64)
        ref=cp.argsort(step.squared(x[:,None].astype(cp.float64),x[None].astype(cp.float64)),axis=1,kind='stable')[:,:120]
        assert bool(cp.all(a==b)) and bool(cp.all(a==ref)), 'smoke neighbors differ'
    rows=[]
    try:
        for dataset,category,samples in [('minishift','capsule',(0,50,129)),('shapenet','vase9',(0,18,36))]:
            args=step.SimpleNamespace(dataset=dataset,device='cuda:0',num_group=4096,group_size=128,
                max_nn=40,use_LFSA=True,use_MSND=True,num_MSND=2,feature='FPFH',level='ALL',vis_save=False)
            ds=step.PatchCore(args=args).get_dataloader(dataset,'test',category,'ALL').dataset
            for sample in samples:
                bench.atomic(OUT/'progress.json',dict(status='running',dataset=dataset,category=category,sample=sample,completed_samples=len(rows)))
                xyz,_,_,path=ds[sample]
                x=cp.asarray(xyz,dtype=cp.float32)
                idx,build=timed(lambda:step.indexing(x,512,50))
                # Full-size warmup for both variants, excluded from timing.
                a=step.step_search(x,idx,512); b=search(fn,x,idx,512)
                cp.cuda.get_current_stream().synchronize()
                assert bool(cp.all(a==b)), 'full-size neighbor mismatch'
                repetitions=[]
                for repeat in range(4):
                    if repeat%2:
                        b,tb=timed(lambda:search(fn,x,idx,512)); a,ta=timed(lambda:step.step_search(x,idx,512))
                    else:
                        a,ta=timed(lambda:step.step_search(x,idx,512)); b,tb=timed(lambda:search(fn,x,idx,512))
                    assert bool(cp.all(a==b))
                    repetitions.append(dict(dataset=dataset,category=category,sample=sample,repeat=repeat,
                        order='trial-first' if repeat%2 else 'baseline-first',baseline_query_ms=ta,trial_query_ms=tb))
                # Identical ordered neighbors with identical descriptor inputs imply
                # identical descriptors; also directly verify the descriptor kernels.
                verification=step.validate(x,a,b,np.asarray(xyz))
                ta=statistics.median(r['baseline_query_ms'] for r in repetitions)
                tb=statistics.median(r['trial_query_ms'] for r in repetitions)
                row=dict(dataset=dataset,category=category,sample=sample,path=str(path),points=len(x),
                    index_build_ms=build,baseline_query_ms=ta,trial_query_ms=tb,delta_query_ms=tb-ta,
                    delta_query_percent=100*(tb/ta-1),baseline_index_query_ms=build+ta,
                    trial_index_query_ms=build+tb,**verification)
                rows.append(row)
                bench.writecsv(OUT/'samples.csv',rows)
                folder=OUT/f'{dataset}_{category}_{sample}';folder.mkdir(exist_ok=True)
                bench.writecsv(folder/'repeats.csv',repetitions)
                print(row,flush=True)
                del a,b,x,idx
        faster=all(r['trial_query_ms']<r['baseline_query_ms'] for r in rows)
        bench.atomic(OUT/'summary.json',dict(status='complete',samples=len(rows),
            all_neighbors_identical=all(r['order_diff']==0 for r in rows),
            all_fpfh_identical=all(r[f'fpfh{k}_diff_rows']==0 for r in rows for k in (40,80,120)),
            faster_every_sample=faster,next_decision='end-to-end pilot' if faster else 'reject universal replacement; inspect traversal work'))
        with REPORT.open('a',encoding='utf-8') as f:
            f.write('\n## 实測結果（연산 시험）\n\n'.replace('实測結果','실측 결과'))
            f.write('| 데이터셋/물체 | 샘플 | 기존 조회 (ms) | 시험 조회 (ms) | Δ (%) | 이웃 순서 불일치 |\n|---|---:|---:|---:|---:|---:|\n')
            for r in rows:
                f.write(f"| {r['dataset']}/{r['category']} | {r['sample']} | {r['baseline_query_ms']:.3f} | {r['trial_query_ms']:.3f} | {r['delta_query_percent']:+.2f} | {r['order_diff']} |\n")
            f.write('\n- 현재 판단: '+('전체 파이프라인 소규모 비교로 진행할 후보임.' if faster else '모든 샘플에서 빨라지지 않아 일괄 대체는 보류함. 방문 순서 변경에 따른 추가 탐색 비용을 먼저 진단함.')+'\n')
            f.write('- 이 표는 동일 인덱스를 공유한 조회 연산 시험이며 전체 추론 개선이나 AUROC/AP 유지를 직접 검증한 결과는 아님.\n')
        bench.atomic(OUT/'progress.json',dict(status='complete',completed_samples=len(rows)))
    except BaseException:
        import traceback
        bench.atomic(OUT/'progress.json',dict(status='failed',completed_samples=len(rows),error=traceback.format_exc()))
        raise


if __name__=='__main__': main()
