"""Test CUB's stable block radix sort for STEP bounds, not candidate merging."""
import statistics
import traceback
from pathlib import Path
import cupy as cp
import numpy as np
import torch
import run_simple3d_capsule_flashfps as bench
import measure_simple3d_capsule_step as step
from pilot_simple3d_step_no_boundsort import timed, search
from analyze_simple3d_post_top3 import ROOT, REPORT

OUT=ROOT/'radix_pilot'


def kernel():
    code=(bench.HERE/'simple3d_step_index.cu').read_text()
    code='#include <cub/block/block_radix_sort.cuh>\n'+code
    old='    sort_pairs(bound,order,p);'
    new='''    using BoundSort=cub::BlockRadixSort<double,128,4,int>;
    __shared__ typename BoundSort::TempStorage bound_storage;
    double local_keys[4]; int local_values[4];
    #pragma unroll
    for(int j=0;j<4;++j) {
        int at=t*4+j;
        local_keys[j]=at<p?bound[at]:1.0/0.0;
        local_values[j]=at;
    }
    BoundSort(bound_storage).Sort(local_keys,local_values);
    #pragma unroll
    for(int j=0;j<4;++j) {
        int at=t*4+j;
        if(at<p) {bound[at]=local_keys[j];order[at]=local_values[j];}
    }
    __syncthreads();'''
    assert code.count(old)==1
    code=code.replace(old,new)
    return cp.RawKernel(code,'search',options=('--std=c++17','--fmad=false',
        '-I/home/test/simple3d-native-toolkit/targets/x86_64-linux/include'))


def main():
    if OUT.exists(): raise RuntimeError('Existing radix pilot preserved')
    OUT.mkdir(parents=True)
    torch.set_num_threads(4)
    bench.atomic(OUT/'plan.json',dict(question='Can stable CUB radix sort reduce bound sorting cost without changing visitation order?',
        hypothesis='Same ordered bounds, 120-NN and FPFH; query time may decrease, but shared-memory/register pressure may refute this',
        fixed='Same input/index, STEP512/Lloyd50, Float64 bounds, existing bitonic candidate merge, all other stages unchanged',
        source_sha256={str(p):bench.sha(p) for p in (Path(__file__),bench.HERE/'simple3d_step_index.cu')},
        library='installed NVIDIA CUB BlockRadixSort<double,128,4,int>, stable blocked ordering',
        library_header_sha256=bench.sha(Path('/home/test/simple3d-native-toolkit/targets/x86_64-linux/include/cub/block/block_radix_sort.cuh')),
        targets={'minishift/capsule':[0,50,129],'shapenet/vase9':[0,18,36]},repeats=4))
    rows=[]
    try:
        fn=kernel()
        for tied in (False,True):
            x=cp.random.RandomState(81).random_sample((1024,3),dtype=cp.float32)
            if tied:x[1:8]=x[0]
            index=step.indexing(x,64,5)
            a=step.step_search(x,index,64);b=search(fn,x,index,64)
            ref=cp.argsort(step.squared(x[:,None].astype(cp.float64),x[None].astype(cp.float64)),axis=1,kind='stable')[:,:120]
            assert bool(cp.all(a==b)) and bool(cp.all(a==ref))
        for dataset,category,samples in [('minishift','capsule',(0,50,129)),('shapenet','vase9',(0,18,36))]:
            args=step.SimpleNamespace(dataset=dataset,device='cuda:0',num_group=4096,group_size=128,
                max_nn=40,use_LFSA=True,use_MSND=True,num_MSND=2,feature='FPFH',level='ALL',vis_save=False)
            ds=step.PatchCore(args=args).get_dataloader(dataset,'test',category,'ALL').dataset
            for sample in samples:
                bench.atomic(OUT/'progress.json',dict(status='running',dataset=dataset,category=category,sample=sample,completed_samples=len(rows)))
                xyz,_,_,path=ds[sample];x=cp.asarray(xyz,dtype=cp.float32)
                index,build=timed(lambda:step.indexing(x,512,50))
                a=step.step_search(x,index,512);b=search(fn,x,index,512)
                cp.cuda.get_current_stream().synchronize();assert bool(cp.all(a==b))
                repeats=[]
                for i in range(4):
                    if i%2:
                        b,tb=timed(lambda:search(fn,x,index,512));a,ta=timed(lambda:step.step_search(x,index,512))
                    else:
                        a,ta=timed(lambda:step.step_search(x,index,512));b,tb=timed(lambda:search(fn,x,index,512))
                    assert bool(cp.all(a==b))
                    repeats.append(dict(dataset=dataset,category=category,sample=sample,repeat=i,
                        order='radix-first' if i%2 else 'baseline-first',baseline_query_ms=ta,radix_query_ms=tb))
                check=step.validate(x,a,b,np.asarray(xyz))
                ta,tb=[statistics.median(r[k] for r in repeats) for k in ('baseline_query_ms','radix_query_ms')]
                row=dict(dataset=dataset,category=category,sample=sample,points=len(x),path=str(path),
                    index_build_ms=build,baseline_query_ms=ta,radix_query_ms=tb,
                    delta_query_percent=100*(tb/ta-1),baseline_index_query_ms=build+ta,
                    radix_index_query_ms=build+tb,**check)
                rows.append(row);bench.writecsv(OUT/'samples.csv',rows)
                folder=OUT/f'{dataset}_{category}_{sample}';folder.mkdir()
                bench.writecsv(folder/'repeats.csv',repeats)
                print(row,flush=True)
                del a,b,index,x
        faster=all(r['radix_query_ms']<r['baseline_query_ms'] for r in rows)
        bench.atomic(OUT/'summary.json',dict(status='complete',samples=len(rows),all_neighbors_identical=True,
            all_fpfh_identical=all(r[f'fpfh{k}_diff_rows']==0 for r in rows for k in (40,80,120)),
            faster_every_sample=faster,next_decision='small end-to-end pilot' if faster else 'reject universal replacement; examine memory/occupancy cost'))
        with REPORT.open('a',encoding='utf-8') as f:
            f.write('\n## 두 번째 후보: 정렬 순서를 유지한 CUB Radix 교체\n\n')
            f.write('정렬 제거 실패를 바탕으로 512개 하한의 정렬만 NVIDIA CUB stable block radix sort로 교체함. 후보 병합 정렬과 탐색 순서·인덱스 구성은 유지함. 기존 라이브러리 교체이며 그 자체를 새로운 방법으로 주장하지 않음.\n\n')
            f.write('| 물체/샘플 | 기존 조회 (ms) | Radix 조회 (ms) | Δ (%) | 이웃 불일치 |\n|---|---:|---:|---:|---:|\n')
            for r in rows:
                f.write(f"| {r['category']}/{r['sample']} | {r['baseline_query_ms']:.3f} | {r['radix_query_ms']:.3f} | {r['delta_query_percent']:+.2f} | {r['order_diff']} |\n")
            f.write('\n- 판단: '+('전체 파이프라인 소규모 비교로 진행할 후보임.' if faster else '모든 입력에서 개선되지 않아 일괄 교체는 보류함.')+'\n')
            f.write('- 동일 인덱스의 조회 연산 시험이므로 아직 전체 추론·AUROC/AP 유지 결과는 아님.\n')
            f.write('- [원본 결과](../source/results/post_top3_bottleneck_20261118/radix_pilot/samples.csv)\n')
        bench.atomic(OUT/'progress.json',dict(status='complete',completed_samples=len(rows)))
    except BaseException:
        bench.atomic(OUT/'progress.json',dict(status='failed',completed_samples=len(rows),error=traceback.format_exc()))
        raise


if __name__=='__main__':main()
