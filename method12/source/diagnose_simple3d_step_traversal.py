"""Untimed traversal counters explain the rejected no-bound-sort candidate."""
from pathlib import Path
import cupy as cp
import numpy as np
import torch
import run_simple3d_capsule_flashfps as bench
import measure_simple3d_capsule_step as step
from pilot_simple3d_step_no_boundsort import kernel, search
from analyze_simple3d_post_top3 import ROOT, REPORT

OUT=ROOT/'traversal_diagnosis'


def instrument(code):
    old='const int* members,int p,int n,int* result) {'
    assert code.count(old)==1
    code=code.replace(old,'const int* members,int p,int n,int* result,unsigned long long* stats) {')
    code=code.replace('int q=blockIdx.x, t=threadIdx.x;',
        'int q=blockIdx.x, t=threadIdx.x; unsigned long long visits=0,candidates=0,merges=0;')
    old='        for (int start=offsets[g];start<offsets[g+1];start+=128) {'
    assert code.count(old)==1
    code=code.replace(old,'        visits++;\n'+old+'\n            candidates+=min(128,offsets[g+1]-start);')
    old='            if(needed) sort_pairs(best,ids,256);'
    assert code.count(old)==1
    code=code.replace(old,'            if(needed) { merges++; sort_pairs(best,ids,256); }')
    code=code.replace('    if(t<120) result[(long long)q*120+t]=ids[t];',
        '    if(t==0) {stats[(long long)q*3]=visits;stats[(long long)q*3+1]=candidates;stats[(long long)q*3+2]=merges;}\n'
        '    if(t<120) result[(long long)q*120+t]=ids[t];')
    return cp.RawKernel(code,'search',options=('--std=c++17','--fmad=false'))


def main():
    if OUT.exists():raise RuntimeError('Existing diagnosis preserved')
    OUT.mkdir(parents=True)
    torch.set_num_threads(4)
    base=instrument((bench.HERE/'simple3d_step_index.cu').read_text())
    trial=instrument(kernel()[0].code)
    bench.atomic(OUT/'plan.json',dict(question='Does unsorted traversal inspect more clusters and points?',
        targets=['minishift/capsule/0','shapenet/vase9/18'],timing='not measured: instrumentation changes execution',
        source_sha256={str(p):bench.sha(p) for p in (Path(__file__),bench.HERE/'pilot_simple3d_step_no_boundsort.py',bench.HERE/'simple3d_step_index.cu')}))
    rows=[]
    for dataset,category,sample in [('minishift','capsule',0),('shapenet','vase9',18)]:
        args=step.SimpleNamespace(dataset=dataset,device='cuda:0',num_group=4096,group_size=128,
            max_nn=40,use_LFSA=True,use_MSND=True,num_MSND=2,feature='FPFH',level='ALL',vis_save=False)
        ds=step.PatchCore(args=args).get_dataloader(dataset,'test',category,'ALL').dataset
        xyz,_,_,path=ds[sample]
        x=cp.asarray(xyz,dtype=cp.float32);idx=step.indexing(x,512,50)
        reference=None
        for name,fn in [('sorted',base),('unsorted',trial)]:
            out=cp.empty((len(x),120),cp.int32);counts=cp.empty((len(x),3),cp.uint64)
            fn((len(x),),(128,),(x,*idx,np.int32(512),np.int32(len(x)),out,counts))
            cp.cuda.get_current_stream().synchronize()
            if reference is None:reference=out
            else:assert bool(cp.all(reference==out))
            values=cp.asnumpy(counts)
            row=dict(dataset=dataset,category=category,sample=sample,variant=name,points=len(x),
                mean_visited_clusters=float(values[:,0].mean()),
                mean_candidate_distances=float(values[:,1].mean()),
                mean_candidate_merges=float(values[:,2].mean()))
            rows.append(row);print(row,flush=True)
            bench.writecsv(OUT/'counters.csv',rows)
        del x,idx,reference,out,counts
    bench.atomic(OUT/'summary.json',dict(status='complete',all_neighbors_identical=True,samples=2))
    with REPORT.open('a',encoding='utf-8') as f:
        f.write('\n## 失敗 원인 계측\n\n'.replace('失敗','실패'))
        f.write('별도 계측 커널로 쿼리당 평균 작업량을 측정함. 계측 오버헤드가 있으므로 이 실행의 시간은 비교하지 않음.\n\n')
        f.write('| 물체/샘플 | 방문 순서 | 방문 군집/쿼리 | 후보 거리 계산/쿼리 | 후보 병합/쿼리 |\n|---|---|---:|---:|---:|\n')
        for r in rows:
            f.write(f"| {r['category']}/{r['sample']} | {r['variant']} | {r['mean_visited_clusters']:.3f} | {r['mean_candidate_distances']:.3f} | {r['mean_candidate_merges']:.3f} |\n")
        f.write('\n- [작업량 원본 CSV](../source/results/post_top3_bottleneck_20261118/traversal_diagnosis/counters.csv)\n')


if __name__=='__main__':main()
